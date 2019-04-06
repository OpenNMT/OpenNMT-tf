"""Standard sequence-to-sequence model."""

import tensorflow as tf

from opennmt import constants
from opennmt import inputters
from opennmt import layers

from opennmt.layers import reducer
from opennmt.models.model import Model
from opennmt.utils import compat
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.misc import print_bytes, format_translation_output, merge_dict, shape_list
from opennmt.decoders.decoder import get_sampling_probability


def shift_target_sequence(inputter, data):
  """Prepares shifted target sequences.

  Given a target sequence ``a b c``, the decoder input should be
  ``<s> a b c`` and the output should be ``a b c </s>`` for the dynamic
  decoding to start on ``<s>`` and stop on ``</s>``.

  Args:
    inputter: The :class:`opennmt.inputters.inputter.Inputter` that processed
      :obj:`data`.
    data: A dict of ``tf.Tensor`` containing ``ids`` and ``length`` keys.

  Returns:
    The updated :obj:`data` dictionary with ``ids`` the sequence prefixed
    with the start token id and ``ids_out`` the sequence suffixed with
    the end token id. Additionally, the ``length`` is increased by 1
    to reflect the added token on both sequences.
  """
  _ = inputter
  ids = data["ids"]
  bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=ids.dtype)
  eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=ids.dtype)
  data["ids_out"] = tf.concat([ids, eos], axis=0)
  data["ids"] = tf.concat([bos, ids], axis=0)
  data["length"] += 1  # Increment length accordingly.
  return data


class EmbeddingsSharingLevel(object):
  """Level of embeddings sharing.

  Possible values are:

   * ``NONE``: no sharing (default)
   * ``SOURCE_TARGET_INPUT``: share source and target word embeddings
   * ``TARGET``: share target word embeddings and softmax weights
   * ``ALL``: share words embeddings and softmax weights
  """
  NONE = 0
  SOURCE_TARGET_INPUT = 1
  TARGET = 2
  ALL = 3

  @staticmethod
  def share_input_embeddings(level):
    """Returns ``True`` if input embeddings should be shared at :obj:`level`."""
    return level in (EmbeddingsSharingLevel.SOURCE_TARGET_INPUT, EmbeddingsSharingLevel.ALL)

  @staticmethod
  def share_target_embeddings(level):
    """Returns ``True`` if target embeddings should be shared at :obj:`level`."""
    return level in (EmbeddingsSharingLevel.TARGET, EmbeddingsSharingLevel.ALL)


class SequenceToSequence(Model):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
               share_embeddings=EmbeddingsSharingLevel.NONE,
               alignment_file_key="train_alignments",
               daisy_chain_variables=False,
               name="seq2seq"):
    """Initializes a sequence-to-sequence model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the source data.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the target data. Currently, only the
        :class:`opennmt.inputters.text_inputter.WordEmbedder` is supported.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the source.
      decoder: A :class:`opennmt.decoders.decoder.Decoder` to decode the target.
      share_embeddings: Level of embeddings sharing, see
        :class:`opennmt.models.sequence_to_sequence.EmbeddingsSharingLevel`
        for possible values.
      alignment_file_key: The data configuration key of the training alignment
        file to support guided alignment.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      TypeError: if :obj:`target_inputter` is not a
        :class:`opennmt.inputters.text_inputter.WordEmbedder` (same for
        :obj:`source_inputter` when embeddings sharing is enabled) or if
        :obj:`source_inputter` and :obj:`target_inputter` do not have the same
        ``dtype``.
    """
    if source_inputter.dtype != target_inputter.dtype:
      raise TypeError(
          "Source and target inputters must have the same dtype, "
          "saw: {} and {}".format(source_inputter.dtype, target_inputter.dtype))
    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")
    if EmbeddingsSharingLevel.share_input_embeddings(share_embeddings):
      if isinstance(source_inputter, inputters.ParallelInputter):
        source_inputters = source_inputter.inputters
      else:
        source_inputters = [source_inputter]
      for inputter in source_inputters:
        if not isinstance(inputter, inputters.WordEmbedder):
          raise TypeError("Sharing embeddings requires all inputters to be a "
                          "WordEmbedder")

    examples_inputter = SequenceToSequenceInputter(
        source_inputter,
        target_inputter,
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings),
        alignment_file_key=alignment_file_key)
    super(SequenceToSequence, self).__init__(
        name,
        daisy_chain_variables=daisy_chain_variables,
        examples_inputter=examples_inputter)

    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.output_layer = None

  def auto_config(self, num_devices=1):
    config = super(SequenceToSequence, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "beam_width": 4,
            "length_penalty": 0.6
        },
        "train": {
            "sample_buffer_size": -1,
            "train_steps": 500000
        },
        "infer": {
            "batch_size": 32,
            "bucket_width": 5
        }
    })

  def _build(self):
    self.examples_inputter.build()
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      self.output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True,
          dtype=self.labels_inputter.dtype)
      with tf.name_scope(tf.get_variable_scope().name + "/"):
        self.output_layer.build([None, self.decoder.output_size])

  def _call(self, features, labels, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN

    features_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter.make_inputs(features, training=training)
    with tf.variable_scope("encoder"):
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
          source_inputs,
          sequence_length=features_length,
          mode=mode)

    target_vocab_size = self.labels_inputter.vocabulary_size
    target_dtype = self.labels_inputter.dtype
    if labels is not None:
      target_inputs = self.labels_inputter.make_inputs(labels, training=training)
      with tf.variable_scope("decoder"):
        sampling_probability = None
        if mode == tf.estimator.ModeKeys.TRAIN:
          sampling_probability = get_sampling_probability(
              tf.train.get_or_create_global_step(),
              read_probability=params.get("scheduled_sampling_read_probability"),
              schedule_type=params.get("scheduled_sampling_type"),
              k=params.get("scheduled_sampling_k"))

        logits, _, _, attention = self.decoder.decode(
            target_inputs,
            self.labels_inputter.get_length(labels),
            vocab_size=target_vocab_size,
            initial_state=encoder_state,
            sampling_probability=sampling_probability,
            embedding=self.labels_inputter.embedding,
            output_layer=self.output_layer,
            mode=mode,
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length,
            return_alignment_history=True)
        if "alignment" in labels:
          outputs = {
              "logits": logits,
              "attention": attention
          }
        else:
          outputs = logits
    else:
      outputs = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope("decoder", reuse=labels is not None):
        batch_size = tf.shape(tf.contrib.framework.nest.flatten(encoder_outputs)[0])[0]
        beam_width = params.get("beam_width", 1)
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID
        sampled_ids, _, sampled_length, log_probs, alignment = (
            self.decoder.dynamic_decode_and_search(
                self.labels_inputter.embedding,
                start_tokens,
                end_token,
                vocab_size=target_vocab_size,
                initial_state=encoder_state,
                output_layer=self.output_layer,
                beam_width=beam_width,
                length_penalty=params.get("length_penalty", 0),
                maximum_iterations=params.get("maximum_iterations", 250),
                minimum_length=params.get("minimum_decoding_length", 0),
                mode=mode,
                memory=encoder_outputs,
                memory_sequence_length=encoder_sequence_length,
                dtype=target_dtype,
                return_alignment_history=True,
                sample_from=params.get("sampling_topk"),
                sample_temperature=params.get("sampling_temperature")))

      target_vocab_rev = self.labels_inputter.vocabulary_lookup_reverse()
      target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))

      if params.get("replace_unknown_target", False):
        if alignment is None:
          raise TypeError("replace_unknown_target is not compatible with decoders "
                          "that don't return alignment history")
        if not isinstance(self.features_inputter, inputters.WordEmbedder):
          raise TypeError("replace_unknown_target is only defined when the source "
                          "inputter is a WordEmbedder")
        source_tokens = features["tokens"]
        if beam_width > 1:
          source_tokens = tf.contrib.seq2seq.tile_batch(source_tokens, multiplier=beam_width)
        # Merge batch and beam dimensions.
        original_shape = tf.shape(target_tokens)
        target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
        align_shape = shape_list(alignment)
        attention = tf.reshape(
            alignment, [align_shape[0] * align_shape[1], align_shape[2], align_shape[3]])
        # We don't have attention for </s> but ensure that the attention time dimension matches
        # the tokens time dimension.
        attention = reducer.align_in_time(attention, tf.shape(target_tokens)[1])
        replaced_target_tokens = replace_unknown_target(target_tokens, source_tokens, attention)
        target_tokens = tf.reshape(replaced_target_tokens, original_shape)

      predictions = {
          "tokens": target_tokens,
          "length": sampled_length,
          "log_probs": log_probs
      }
      if alignment is not None:
        predictions["alignment"] = alignment
    else:
      predictions = None

    return outputs, predictions

  def compute_loss(self, outputs, labels, training=True, params=None):
    if params is None:
      params = {}
    if isinstance(outputs, dict):
      logits = outputs["logits"]
      attention = outputs.get("attention")
    else:
      logits = outputs
      attention = None
    labels_lengths = self.labels_inputter.get_length(labels)
    loss, loss_normalizer, loss_token_normalizer = cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.logging.warning("This model did not return attention vectors; "
                             "guided alignment will not be applied")
        else:
          # Note: the first decoder input is <s> for which we don't want any alignment.
          loss += guided_alignment_cost(
              attention[:, 1:],
              gold_alignments,
              labels_lengths - 1,
              guided_alignment_type,
              guided_alignment_weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    n_best = params and params.get("n_best")
    n_best = n_best or 1

    if n_best > len(prediction["tokens"]):
      raise ValueError("n_best cannot be greater than beam_width")

    for i in range(n_best):
      target_length = prediction["length"][i] - 1  # Ignore </s>.
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params is not None and params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params is not None and params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)


class SequenceToSequenceInputter(inputters.ExampleInputter):
  """A custom :class:`opennmt.inputters.inputter.ExampleInputter` that possibly
  injects alignment information during training.
  """

  def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False,
               alignment_file_key=None):
    super(SequenceToSequenceInputter, self).__init__(
        features_inputter, labels_inputter, share_parameters=share_parameters)
    self.alignment_file_key = alignment_file_key
    self.alignment_file = None

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    if self.alignment_file_key is not None and self.alignment_file_key in metadata:
      self.alignment_file = metadata[self.alignment_file_key]
    return super(SequenceToSequenceInputter, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)

  def make_dataset(self, data_file, training=None):
    dataset = super(SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
    if self.alignment_file is None or not training:
      return dataset
    return tf.data.Dataset.zip((dataset, tf.data.TextLineDataset(self.alignment_file)))

  def make_features(self, element=None, features=None, training=None):
    if self.alignment_file is None or not training:
      return super(SequenceToSequenceInputter, self).make_features(
          element=element, features=features, training=training)
    text, alignment = element
    features, labels = super(SequenceToSequenceInputter, self).make_features(
        text, features=features, training=training)
    labels["alignment"] = alignment_matrix_from_pharaoh(
        alignment,
        self.features_inputter.get_length(features),
        self.labels_inputter.get_length(labels) - 1)  # Ignore special token.
    return features, labels

  def _get_names(self):
    return ["encoder", "decoder"]

  def _get_shared_name(self):
    return "shared_embeddings"


def alignment_matrix_from_pharaoh(alignment_line,
                                  source_length,
                                  target_length,
                                  dtype=tf.float32):
  """Parse Pharaoh alignments into an alignment matrix.

  Args:
    alignment_line: A string ``tf.Tensor`` in the Pharaoh format.
    source_length: The length of the source sentence, without special symbols.
    target_length The length of the target sentence, without special symbols.
    dtype: The output matrix dtype. Defaults to ``tf.float32`` for convenience
      when computing the guided alignment loss.

  Returns:
    The alignment matrix as a 2-D ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[target_length, source_length]``, where ``[i, j] = 1`` if the ``i`` th
    target word is aligned with the ``j`` th source word.
  """
  if compat.tf_supports("strings.split"):
    align_pairs_str = tf.strings.split([alignment_line]).values
    align_pairs_flat_str = tf.strings.split(align_pairs_str, sep="-").values
  else:
    align_pairs_str = tf.string_split([alignment_line], delimiter=" ").values
    align_pairs_flat_str = tf.string_split(align_pairs_str, delimiter="-").values
  align_pairs_flat = compat.tf_compat(v2="strings.to_number", v1="string_to_number")(
      align_pairs_flat_str, out_type=tf.int64)
  sparse_indices = tf.reshape(align_pairs_flat, [-1, 2])
  sparse_values = tf.ones([tf.shape(sparse_indices)[0]], dtype=dtype)
  source_length = tf.cast(source_length, tf.int64)
  target_length = tf.cast(target_length, tf.int64)
  if compat.tf_supports("sparse.to_dense"):
    alignment_matrix_sparse = tf.sparse.SparseTensor(
        sparse_indices, sparse_values, [source_length, target_length])
    alignment_matrix = tf.sparse.to_dense(alignment_matrix_sparse, validate_indices=False)
  else:
    alignment_matrix = tf.sparse_to_dense(
        sparse_indices,
        [source_length, target_length],
        sparse_values,
        validate_indices=False)
  return tf.transpose(alignment_matrix)

def guided_alignment_cost(attention_probs,
                          gold_alignment,
                          sequence_length,
                          guided_alignment_type,
                          guided_alignment_weight=1):
  """Computes the guided alignment cost.

  Args:
    attention_probs: The attention probabilities, a float ``tf.Tensor`` of shape
      :math:`[B, T_t, T_s]`.
    gold_alignment: The true alignment matrix, a float ``tf.Tensor`` of shape
      :math:`[B, T_t, T_s]`.
    sequence_length: The length of each sequence.
    guided_alignment_type: The type of guided alignment cost function to compute
      (can be: ce, mse).
    guided_alignment_weight: The weight applied to the guided alignment cost.

  Returns:
    The guided alignment cost.
  """
  weights = tf.sequence_mask(
      sequence_length, maxlen=tf.shape(attention_probs)[1], dtype=attention_probs.dtype)
  if guided_alignment_type == "ce":
    cross_entropy = -tf.reduce_sum(tf.log(attention_probs + 1e-6) * gold_alignment, axis=-1)
    loss = tf.reduce_sum(cross_entropy * weights)
  elif guided_alignment_type == "mse":
    loss = tf.losses.mean_squared_error(
        gold_alignment, attention_probs, weights=tf.expand_dims(weights, -1))
  else:
    raise ValueError("invalid guided_alignment_type: %s" % guided_alignment_type)
  return guided_alignment_weight * loss

def align_tokens_from_attention(tokens, attention):
  """Returns aligned tokens from the attention.

  Args:
    tokens: The tokens on which the attention is applied as a string
      ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.

  Returns:
    The aligned tokens as a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
  """
  alignment = tf.argmax(attention, axis=-1, output_type=tf.int32)
  batch_size = tf.shape(tokens)[0]
  max_time = tf.shape(attention)[1]
  batch_ids = tf.range(batch_size)
  batch_ids = tf.tile(batch_ids, [max_time])
  batch_ids = tf.reshape(batch_ids, [max_time, batch_size])
  batch_ids = tf.transpose(batch_ids, perm=[1, 0])
  aligned_pos = tf.stack([batch_ids, alignment], axis=-1)
  aligned_tokens = tf.gather_nd(tokens, aligned_pos)
  return aligned_tokens

def replace_unknown_target(target_tokens,
                           source_tokens,
                           attention,
                           unknown_token=constants.UNKNOWN_TOKEN):
  """Replaces all target unknown tokens by the source token with the highest
  attention.

  Args:
    target_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_t]`.
    source_tokens: A a string ``tf.Tensor`` of shape :math:`[B, T_s]`.
    attention: The attention vector of shape :math:`[B, T_t, T_s]`.
    unknown_token: The target token to replace.

  Returns:
    A string ``tf.Tensor`` with the same shape and type as :obj:`target_tokens`
    but will all instances of :obj:`unknown_token` replaced by the aligned source
    token.
  """
  aligned_source_tokens = align_tokens_from_attention(source_tokens, attention)
  return tf.where(
      tf.equal(target_tokens, unknown_token),
      x=aligned_source_tokens,
      y=target_tokens)
