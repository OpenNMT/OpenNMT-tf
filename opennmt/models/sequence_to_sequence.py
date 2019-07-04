# -*- coding: utf-8 -*-

"""Standard sequence-to-sequence model."""

import six

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import constants
from opennmt import inputters
from opennmt import layers

from opennmt.data import noise
from opennmt.data import vocab
from opennmt.layers import reducer
from opennmt.models.model import Model
from opennmt.utils import decoding
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.misc import print_bytes, format_translation_output, merge_dict, shape_list
from opennmt.decoders import decoder as decoder_util


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
               share_embeddings=EmbeddingsSharingLevel.NONE):
    """Initializes a sequence-to-sequence model.

    Args:
      source_inputter: A :class:`opennmt.inputters.Inputter` to process
        the source data.
      target_inputter: A :class:`opennmt.inputters.Inputter` to process
        the target data. Currently, only the
        :class:`opennmt.inputters.WordEmbedder` is supported.
      encoder: A :class:`opennmt.encoders.Encoder` to encode the source.
      decoder: A :class:`opennmt.decoders.Decoder` to decode the target.
      share_embeddings: Level of embeddings sharing, see
        :class:`opennmt.models.EmbeddingsSharingLevel`
        for possible values.
      name: The name of this model.

    Raises:
      TypeError: if :obj:`target_inputter` is not a
        :class:`opennmt.inputters.WordEmbedder` (same for
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
        share_parameters=EmbeddingsSharingLevel.share_input_embeddings(share_embeddings))
    super(SequenceToSequence, self).__init__(examples_inputter)
    self.encoder = encoder
    self.decoder = decoder
    self.share_embeddings = share_embeddings

  def auto_config(self, num_replicas=1):
    config = super(SequenceToSequence, self).auto_config(num_replicas=num_replicas)
    return merge_dict(config, {
        "params": {
            "beam_width": 4
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

  def build(self, input_shape):
    super(SequenceToSequence, self).build(input_shape)
    output_layer = None
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      output_layer = layers.Dense(
          self.labels_inputter.vocabulary_size,
          weight=self.labels_inputter.embedding,
          transpose=True,
          dtype=self.labels_inputter.dtype)
    self.decoder.initialize(
        vocab_size=self.labels_inputter.vocabulary_size,
        output_layer=output_layer)
    self.id_to_token = None

  def call(self, features, labels=None, step=None, mode=tf.estimator.ModeKeys.PREDICT):
    params = self.params
    training = mode == tf.estimator.ModeKeys.TRAIN

    features_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs,
        sequence_length=features_length,
        training=training)

    if labels is not None:
      target_inputs = self.labels_inputter(labels, training=training)
      sampling_probability = None
      if mode == tf.estimator.ModeKeys.TRAIN:
        sampling_probability = decoder_util.get_sampling_probability(
            step,
            read_probability=params.get("scheduled_sampling_read_probability"),
            schedule_type=params.get("scheduled_sampling_type"),
            k=params.get("scheduled_sampling_k"))

      initial_state = self.decoder.initial_state(
          memory=encoder_outputs,
          memory_sequence_length=encoder_sequence_length,
          initial_state=encoder_state)
      logits, _, attention = self.decoder(
          target_inputs,
          self.labels_inputter.get_length(labels),
          state=initial_state,
          input_fn=lambda ids: self.labels_inputter({"ids": ids}, training=training),
          sampling_probability=sampling_probability,
          training=training)
      outputs = dict(logits=logits, attention=attention)
    else:
      outputs = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
      start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
      beam_size = params.get("beam_width", 1)
      if beam_size > 1:
        encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
        encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
        if encoder_state is not None:
          encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)
      initial_state = self.decoder.initial_state(
          memory=encoder_outputs,
          memory_sequence_length=encoder_sequence_length,
          initial_state=encoder_state)
      sampled_ids, sampled_length, log_probs, alignment, _ = decoding.dynamic_decode_from_params(
          self.decoder,
          self.labels_inputter,
          start_ids,
          initial_state=initial_state,
          params=params)
      if self.id_to_token is None:
        with tf.init_scope():
          self.id_to_token = self.labels_inputter.vocabulary_lookup_reverse()
      target_tokens = self.id_to_token.lookup(tf.cast(sampled_ids, tf.int64))

      if params.get("replace_unknown_target", False):
        if alignment is None:
          raise TypeError("replace_unknown_target is not compatible with decoders "
                          "that don't return alignment history")
        if not isinstance(self.features_inputter, inputters.WordEmbedder):
          raise TypeError("replace_unknown_target is only defined when the source "
                          "inputter is a WordEmbedder")
        source_tokens = features["tokens"]
        if beam_size > 1:
          source_tokens = tfa.seq2seq.tile_batch(source_tokens, beam_size)
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

      decoding_noise = params.get("decoding_noise")
      if decoding_noise:
        target_tokens, sampled_length = _add_noise(
            target_tokens,
            sampled_length,
            decoding_noise,
            params.get("decoding_subword_token", "￭"))
        alignment = None  # Invalidate alignments.

      predictions = {
          "tokens": target_tokens,
          "length": sampled_length,
          "log_probs": log_probs
      }
      if alignment is not None:
        predictions["alignment"] = alignment

      num_hypotheses = params.get("num_hypotheses", 1)
      if num_hypotheses > 0:
        if num_hypotheses > beam_size:
          raise ValueError("n_best cannot be greater than beam_width")
        for key, value in six.iteritems(predictions):
          predictions[key] = value[:, :num_hypotheses]
    else:
      predictions = None

    return outputs, predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
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
          tf.compat.v1.logging.warning("This model did not return attention vectors; "
                                       "guided alignment will not be applied")
        else:
          loss += guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              labels_lengths - 1,
              guided_alignment_type,
              guided_alignment_weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    num_hypotheses = len(prediction["tokens"])
    for i in range(num_hypotheses):
      target_length = prediction["length"][i]
      tokens = prediction["tokens"][i][:target_length]
      sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      score = None
      attention = None
      alignment_type = None
      if params.get("with_scores"):
        score = prediction["log_probs"][i]
      if params.get("with_alignments"):
        attention = prediction["alignment"][i][:target_length]
        alignment_type = params["with_alignments"]
      sentence = format_translation_output(
          sentence,
          score=score,
          attention=attention,
          alignment_type=alignment_type)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
    updated_variables = []

    def _map_variables(inputter_fn, vars_fn):
      mapping, _ = vocab.get_mapping(
          inputter_fn(self).vocabulary_file,
          inputter_fn(new_model).vocabulary_file)
      vars_a, vocab_axes = vars_fn(self)
      vars_b, _ = vars_fn(new_model)
      for var_a, var_b, vocab_axis in zip(vars_a, vars_b, vocab_axes):
        if new_optimizer is not None and optimizer is not None:
          variables = vocab.update_variable_and_slots(
              var_a,
              var_b,
              optimizer,
              new_optimizer,
              mapping,
              vocab_axis=vocab_axis)
        else:
          variables = [vocab.update_variable(var_a, var_b, mapping, vocab_axis=vocab_axis)]
        updated_variables.extend(variables)
      return vars_b

    _map_variables(
        lambda model: model.features_inputter,
        lambda model: ([model.features_inputter.embedding], [0]))
    _map_variables(
        lambda model: model.labels_inputter,
        lambda model: ([
            model.labels_inputter.embedding,
            model.decoder.output_layer.kernel,
            model.decoder.output_layer.bias], [0, 1, 0]))

    return super(SequenceToSequence, self).transfer_weights(
        new_model,
        new_optimizer=new_optimizer,
        optimizer=optimizer,
        ignore_weights=set(updated_variables))


class SequenceToSequenceInputter(inputters.ExampleInputter):
  """A custom :class:`opennmt.inputters.ExampleInputter` that possibly
  injects alignment information during training.
  """

  def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
    super(SequenceToSequenceInputter, self).__init__(
        features_inputter, labels_inputter, share_parameters=share_parameters)
    self.alignment_file = None

  def initialize(self, data_config, asset_prefix=""):
    super(SequenceToSequenceInputter, self).initialize(data_config, asset_prefix=asset_prefix)
    self.alignment_file = data_config.get("train_alignments")

  def make_dataset(self, data_file, training=None):
    dataset = super(SequenceToSequenceInputter, self).make_dataset(
        data_file, training=training)
    if self.alignment_file is None or not training:
      return dataset
    return tf.data.Dataset.zip((dataset, tf.data.TextLineDataset(self.alignment_file)))

  def make_features(self, element=None, features=None, training=None):
    if training and self.alignment_file is not None:
      element, alignment = element
    else:
      alignment = None
    features, labels = super(SequenceToSequenceInputter, self).make_features(
        element=element, features=features, training=training)
    if alignment is not None:
      labels["alignment"] = alignment_matrix_from_pharaoh(
          alignment,
          self.features_inputter.get_length(features),
          self.labels_inputter.get_length(labels))
    labels_ids = labels["ids"]
    bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=labels_ids.dtype)
    eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=labels_ids.dtype)
    labels["ids"] = tf.concat([bos, labels_ids], axis=0)
    labels["ids_out"] = tf.concat([labels_ids, eos], axis=0)
    labels["length"] += 1
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
  align_pairs_str = tf.strings.split([alignment_line]).values
  align_pairs_flat_str = tf.strings.split(align_pairs_str, sep="-").values
  align_pairs_flat = tf.strings.to_number(align_pairs_flat_str, out_type=tf.int64)
  sparse_indices = tf.reshape(align_pairs_flat, [-1, 2])
  sparse_values = tf.ones([tf.shape(sparse_indices)[0]], dtype=dtype)
  source_length = tf.cast(source_length, tf.int64)
  target_length = tf.cast(target_length, tf.int64)
  alignment_matrix_sparse = tf.sparse.SparseTensor(
      sparse_indices, sparse_values, [source_length, target_length])
  alignment_matrix = tf.sparse.to_dense(alignment_matrix_sparse, validate_indices=False)
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
    cross_entropy = -tf.reduce_sum(tf.math.log(attention_probs + 1e-6) * gold_alignment, axis=-1)
    loss = tf.reduce_sum(cross_entropy * weights)
  elif guided_alignment_type == "mse":
    loss = tf.keras.losses.MeanSquaredError()(
        gold_alignment, attention_probs, sample_weight=tf.expand_dims(weights, -1))
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

def _add_noise(tokens, lengths, params, subword_token):
  if not isinstance(params, list):
    raise ValueError("Expected a list of noise modules")
  noises = []
  for module in params:
    noise_type, args = six.next(six.iteritems(module))
    if not isinstance(args, list):
      args = [args]
    noise_type = noise_type.lower()
    if noise_type == "dropout":
      noise_class = noise.WordDropout
    elif noise_type == "replacement":
      noise_class = noise.WordReplacement
    elif noise_type == "permutation":
      noise_class = noise.WordPermutation
    else:
      raise ValueError("Invalid noise type: %s" % noise_type)
    noises.append(noise_class(*args))
  noiser = noise.WordNoiser(noises=noises, subword_token=subword_token)
  return noiser(tokens, lengths, keep_shape=True)
