"""Standard sequence-to-sequence model."""

import tensorflow as tf
import numpy as np

import opennmt.constants as constants
import opennmt.inputters as inputters

from opennmt.models.model import Model
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.misc import print_bytes
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
  bos = tf.cast(tf.constant([constants.START_OF_SENTENCE_ID]), tf.int64)
  eos = tf.cast(tf.constant([constants.END_OF_SENTENCE_ID]), tf.int64)

  ids = data["ids"]
  length = data["length"]

  data = inputter.set_data_field(data, "ids_out", tf.concat([ids, eos], axis=0))
  data = inputter.set_data_field(data, "ids", tf.concat([bos, ids], axis=0))

  # Increment length accordingly.
  inputter.set_data_field(data, "length", length + 1)

  return data

def _maybe_reuse_embedding_fn(embedding_fn, scope=None):
  def _scoped_embedding_fn(ids):
    try:
      with tf.variable_scope(scope):
        return embedding_fn(ids)
    except ValueError:
      with tf.variable_scope(scope, reuse=True):
        return embedding_fn(ids)
  return _scoped_embedding_fn


class EmbeddingsSharingLevel(object):
  """Level of embeddings sharing.

  Possible values are:

   * ``NONE``: no sharing (default)
   * ``SOURCE_TARGET_INPUT``: share source and target word embeddings
  """
  NONE = 0
  SOURCE_TARGET_INPUT = 1


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
    if share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      if not isinstance(source_inputter, inputters.WordEmbedder):
        raise TypeError("Sharing embeddings requires both inputters to be a "
                        "WordEmbedder")

    super(SequenceToSequence, self).__init__(
        name,
        features_inputter=source_inputter,
        labels_inputter=target_inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings
    self.source_inputter = source_inputter
    self.target_inputter = target_inputter
    self.target_inputter.add_process_hooks([shift_target_sequence])
    self.alignment_file_key = alignment_file_key
    self.alignment_file = None

  def _initialize(self, metadata):
    super(SequenceToSequence, self)._initialize(metadata)
    if self.alignment_file_key is not None and self.alignment_file_key in metadata:
      self.alignment_file = metadata[self.alignment_file_key]
      tf.logging.info(
          "Training with guided alignment using alignments from %s", self.alignment_file)

  def _augment_parallel_dataset(self, dataset, process_fn, mode=None):
    # Possibly add alignments as labels.
    if self.alignment_file is None or mode != tf.estimator.ModeKeys.TRAIN:
      return dataset, process_fn

    def _inject_alignments(text, alignment_line):
      source, target = text
      features, labels = process_fn(source, target)  # Default processing.
      alignments = alignment_matrix_from_pharaoh(
          alignment_line,
          self._get_features_length(features),
          self._get_labels_length(labels))
      labels["alignment"] = alignments
      return features, labels

    alignment_dataset = tf.data.TextLineDataset(self.alignment_file)
    dataset = tf.data.Dataset.zip((dataset, alignment_dataset))
    return dataset, _inject_alignments

  def _get_input_scope(self, default_name=""):
    if self.share_embeddings == EmbeddingsSharingLevel.SOURCE_TARGET_INPUT:
      name = "shared_embeddings"
    else:
      name = default_name
    return tf.VariableScope(None, name=tf.get_variable_scope().name + "/" + name)

  def _build(self, features, labels, params, mode, config=None):
    features_length = self._get_features_length(features)
    log_dir = config.model_dir if config is not None else None

    source_input_scope = self._get_input_scope(default_name="encoder")
    target_input_scope = self._get_input_scope(default_name="decoder")

    source_inputs = _maybe_reuse_embedding_fn(
        lambda ids: self.source_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
        scope=source_input_scope)(features)

    with tf.variable_scope("encoder"):
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
          source_inputs,
          sequence_length=features_length,
          mode=mode)

    target_vocab_size = self.target_inputter.vocabulary_size
    target_dtype = self.target_inputter.dtype
    target_embedding_fn = _maybe_reuse_embedding_fn(
        lambda ids: self.target_inputter.transform(ids, mode=mode),
        scope=target_input_scope)

    if labels is not None:
      target_inputs = _maybe_reuse_embedding_fn(
          lambda ids: self.target_inputter.transform_data(ids, mode=mode, log_dir=log_dir),
          scope=target_input_scope)(labels)

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
            self._get_labels_length(labels),
            vocab_size=target_vocab_size,
            initial_state=encoder_state,
            sampling_probability=sampling_probability,
            embedding=target_embedding_fn,
            mode=mode,
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length,
            return_alignment_history=True)
        outputs = {
            "logits": logits,
            "attention": attention
        }
    else:
      outputs = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope("decoder", reuse=labels is not None):
        batch_size = tf.shape(encoder_sequence_length)[0]
        beam_width = params.get("beam_width", 1)
        maximum_iterations = params.get("maximum_iterations", 250)
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID

        if beam_width <= 1:
          sampled_ids, _, sampled_length, log_probs, alignment = self.decoder.dynamic_decode(
              target_embedding_fn,
              start_tokens,
              end_token,
              vocab_size=target_vocab_size,
              initial_state=encoder_state,
              maximum_iterations=maximum_iterations,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length,
              dtype=target_dtype,
              return_alignment_history=True)
        else:
          length_penalty = params.get("length_penalty", 0)
          sampled_ids, _, sampled_length, log_probs, alignment = (
              self.decoder.dynamic_decode_and_search(
                  target_embedding_fn,
                  start_tokens,
                  end_token,
                  vocab_size=target_vocab_size,
                  initial_state=encoder_state,
                  beam_width=beam_width,
                  length_penalty=length_penalty,
                  maximum_iterations=maximum_iterations,
                  mode=mode,
                  memory=encoder_outputs,
                  memory_sequence_length=encoder_sequence_length,
                  dtype=target_dtype,
                  return_alignment_history=True))

      target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.target_inputter.vocabulary_file,
          vocab_size=target_vocab_size - self.target_inputter.num_oov_buckets,
          default_value=constants.UNKNOWN_TOKEN)
      target_tokens = target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64))

      if params.get("replace_unknown_target", False):
        if alignment is None:
          raise TypeError("replace_unknown_target is not compatible with decoders "
                          "that don't return alignment history")
        if not isinstance(self.source_inputter, inputters.WordEmbedder):
          raise TypeError("replace_unknown_target is only defined when the source "
                          "inputter is a WordEmbedder")
        source_tokens = features["tokens"]
        if beam_width > 1:
          source_tokens = tf.contrib.seq2seq.tile_batch(source_tokens, multiplier=beam_width)
        # Merge batch and beam dimensions.
        original_shape = tf.shape(target_tokens)
        target_tokens = tf.reshape(target_tokens, [-1, original_shape[-1]])
        attention = tf.reshape(alignment, [-1, tf.shape(alignment)[2], tf.shape(alignment)[3]])
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

  def _compute_loss(self, features, labels, outputs, params, mode):
    if isinstance(outputs, dict):
      logits = outputs["logits"]
      attention = outputs.get("attention")
    else:
      logits = outputs
      attention = None
    labels_lengths = self._get_labels_length(labels)
    loss, loss_normalizer, loss_token_normalizer = cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels_lengths,
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        mode=mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.logging.warning("This model did not return attention vectors; "
                             "guided alignment will not be applied")
        else:
          loss += guided_alignment_cost(
              attention,
              gold_alignments,
              labels_lengths,
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
      sentence = self.target_inputter.tokenizer.detokenize(tokens)
      if params is not None and params.get("with_scores"):
        sentence = "%f ||| %s" % (
            prediction["log_probs"][i] / prediction["length"][i], sentence)
      if params is not None and params.get("with_alignments") == "hard":
        source_indices = np.argmax(prediction["alignment"][i][:target_length], axis=-1)
        target_indices = range(target_length)
        pairs = ("%d-%d" % (src, tgt) for src, tgt in zip(source_indices, target_indices))
        sentence = "%s ||| %s" % (sentence, " ".join(pairs))
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)


def alignment_matrix_from_pharaoh(alignment_line,
                                  source_length,
                                  target_length,
                                  dtype=tf.float32):
  """Parse Pharaoh alignments into an alignment matrix.

  Args:
    alignment_line: A string ``tf.Tensor`` in the Pharaoh format.
    source_length: The length of the source sentence.
    target_length The length of the target sentence.
    dtype: The output matrix dtype. Defaults to ``tf.float32`` for convenience
      when computing the guided alignment loss.

  Returns:
    The alignment matrix as a 2-D ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[target_length, source_length]``, where ``[i, j] = 1`` if the ``i`` th
    target word is aligned with the ``j`` th source word.
  """
  align_pairs_str = tf.string_split([alignment_line], delimiter=" ").values
  align_pairs_flat_str = tf.string_split(align_pairs_str, delimiter="-").values
  align_pairs_flat = tf.string_to_number(align_pairs_flat_str, out_type=tf.int32)
  sparse_indices = tf.reshape(align_pairs_flat, [-1, 2])
  sparse_values = tf.ones([tf.shape(sparse_indices)[0]])
  alignment_matrix = tf.sparse_to_dense(
      sparse_indices,
      [source_length, target_length],
      sparse_values,
      validate_indices=False)
  return tf.cast(tf.transpose(alignment_matrix), dtype)

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
