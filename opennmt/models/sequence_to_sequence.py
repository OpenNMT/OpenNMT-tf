# -*- coding: utf-8 -*-

"""Standard sequence-to-sequence model."""

import six

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import constants
from opennmt import inputters

from opennmt.data import noise
from opennmt.data import text
from opennmt.data import vocab
from opennmt.layers import reducer
from opennmt.models import model
from opennmt.utils import decoding
from opennmt.utils import losses
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


class SequenceToSequence(model.SequenceGenerator):
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

    Raises:
      TypeError: if :obj:`target_inputter` is not a
        :class:`opennmt.inputters.WordEmbedder` (same for
        :obj:`source_inputter` when embeddings sharing is enabled) or if
        :obj:`source_inputter` and :obj:`target_inputter` do not have the same
        ``dtype``.
    """
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
            "max_step": 500000
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 5
        }
    })

  def initialize(self, data_config, params=None):
    super(SequenceToSequence, self).initialize(data_config, params=params)
    self.decoder.initialize(vocab_size=self.labels_inputter.vocabulary_size)
    if self.params.get("contrastive_learning"):
      # Use the simplest and most effective CL_one from the paper.
      # https://www.aclweb.org/anthology/P19-1623
      noiser = noise.WordNoiser(
          noises=[noise.WordOmission(1)],
          subword_token=self.params.get("decoding_subword_token", "￭"),
          is_spacer=self.params.get("decoding_subword_token_is_spacer"))
      self.labels_inputter.set_noise(noiser, in_place=False)

  def build(self, input_shape):
    super(SequenceToSequence, self).build(input_shape)
    if EmbeddingsSharingLevel.share_target_embeddings(self.share_embeddings):
      self.decoder.reuse_embeddings(self.labels_inputter.embedding)

  def call(self, features, labels=None, training=None, step=None):
    # Encode the source.
    source_length = self.features_inputter.get_length(features)
    source_inputs = self.features_inputter(features, training=training)
    encoder_outputs, encoder_state, encoder_sequence_length = self.encoder(
        source_inputs, sequence_length=source_length, training=training)

    outputs = None
    predictions = None

    # When a target is provided, compute the decoder outputs for it.
    if labels is not None:
      outputs = self._decode_target(
          labels,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length,
          step=step,
          training=training)

    # When not in training, also compute the model predictions.
    if not training:
      predictions = self._dynamic_decode(
          features,
          encoder_outputs,
          encoder_state,
          encoder_sequence_length)

    return outputs, predictions

  def _decode_target(self,
                     labels,
                     encoder_outputs,
                     encoder_state,
                     encoder_sequence_length,
                     step=None,
                     training=None):
    params = self.params
    target_inputs = self.labels_inputter(labels, training=training)
    input_fn = lambda ids: self.labels_inputter({"ids": ids}, training=training)

    sampling_probability = None
    if training:
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
        input_fn=input_fn,
        sampling_probability=sampling_probability,
        training=training)
    outputs = dict(logits=logits, attention=attention)

    noisy_ids = labels.get("noisy_ids")
    if noisy_ids is not None and params.get("contrastive_learning"):
      # In case of contrastive learning, also forward the erroneous
      # translation to compute its log likelihood later.
      noisy_inputs = self.labels_inputter({"ids": noisy_ids}, training=training)
      noisy_logits, _, _ = self.decoder(
          noisy_inputs,
          labels["noisy_length"],
          state=initial_state,
          input_fn=input_fn,
          sampling_probability=sampling_probability,
          training=training)
      outputs["noisy_logits"] = noisy_logits
    return outputs

  def _dynamic_decode(self, features, encoder_outputs, encoder_state, encoder_sequence_length):
    params = self.params
    batch_size = tf.shape(tf.nest.flatten(encoder_outputs)[0])[0]
    start_ids = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
    beam_size = params.get("beam_width", 1)

    if beam_size > 1:
      # Tile encoder outputs to prepare for beam search.
      encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
      encoder_sequence_length = tfa.seq2seq.tile_batch(encoder_sequence_length, beam_size)
      if encoder_state is not None:
        encoder_state = tfa.seq2seq.tile_batch(encoder_state, beam_size)

    # Dynamically decodes from the encoder outputs.
    initial_state = self.decoder.initial_state(
        memory=encoder_outputs,
        memory_sequence_length=encoder_sequence_length,
        initial_state=encoder_state)
    sampled_ids, sampled_length, log_probs, alignment, _ = self.decoder.dynamic_decode(
        self.labels_inputter,
        start_ids,
        initial_state=initial_state,
        decoding_strategy=decoding.DecodingStrategy.from_params(params),
        sampler=decoding.Sampler.from_params(params),
        maximum_iterations=params.get("maximum_decoding_length", 250),
        minimum_iterations=params.get("minimum_decoding_length", 0))
    target_tokens = self.labels_inputter.ids_to_tokens.lookup(tf.cast(sampled_ids, tf.int64))

    # Maybe replace unknown targets by the source tokens with the highest attention weight.
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

    # Maybe add noise to the predictions.
    decoding_noise = params.get("decoding_noise")
    if decoding_noise:
      target_tokens, sampled_length = _add_noise(
          target_tokens,
          sampled_length,
          decoding_noise,
          params.get("decoding_subword_token", "￭"),
          params.get("decoding_subword_token_is_spacer"))
      alignment = None  # Invalidate alignments.

    predictions = {"log_probs": log_probs}
    if self.labels_inputter.tokenizer.in_graph:
      detokenized_text = self.labels_inputter.tokenizer.detokenize(
          tf.reshape(target_tokens, [batch_size * beam_size, -1]),
          sequence_length=tf.reshape(sampled_length, [batch_size * beam_size]))
      predictions["text"] = tf.reshape(detokenized_text, [batch_size, beam_size])
    else:
      predictions["tokens"] = target_tokens
      predictions["length"] = sampled_length
      if alignment is not None:
        predictions["alignment"] = alignment

    # Maybe restrict the number of returned hypotheses based on the user parameter.
    num_hypotheses = params.get("num_hypotheses", 1)
    if num_hypotheses > 0:
      if num_hypotheses > beam_size:
        raise ValueError("n_best cannot be greater than beam_width")
      for key, value in six.iteritems(predictions):
        predictions[key] = value[:, :num_hypotheses]
    return predictions

  def compute_loss(self, outputs, labels, training=True):
    params = self.params
    if not isinstance(outputs, dict):
      outputs = dict(logits=outputs)
    logits = outputs["logits"]
    noisy_logits = outputs.get("noisy_logits")
    attention = outputs.get("attention")
    if noisy_logits is not None and params.get("contrastive_learning"):
      return losses.max_margin_loss(
          logits,
          labels["ids_out"],
          labels["length"],
          noisy_logits,
          labels["noisy_ids_out"],
          labels["noisy_length"],
          eta=params.get("max_margin_eta", 0.1))
    loss, loss_normalizer, loss_token_normalizer = losses.cross_entropy_sequence_loss(
        logits,
        labels["ids_out"],
        labels["length"],
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)
    if training:
      gold_alignments = labels.get("alignment")
      guided_alignment_type = params.get("guided_alignment_type")
      if gold_alignments is not None and guided_alignment_type is not None:
        if attention is None:
          tf.get_logger().warning("This model did not return attention vectors; "
                                  "guided alignment will not be applied")
        else:
          loss += losses.guided_alignment_cost(
              attention[:, :-1],  # Do not constrain last timestep.
              gold_alignments,
              sequence_length=self.labels_inputter.get_length(labels, ignore_special_tokens=True),
              cost_type=guided_alignment_type,
              weight=params.get("guided_alignment_weight", 1))
    return loss, loss_normalizer, loss_token_normalizer

  def print_prediction(self, prediction, params=None, stream=None):
    if params is None:
      params = {}
    with_scores = params.get("with_scores")
    alignment_type = params.get("with_alignments")
    if alignment_type and "alignment" not in prediction:
      raise ValueError("with_alignments is set but the model did not return alignment information")
    num_hypotheses = len(prediction["log_probs"])
    for i in range(num_hypotheses):
      if "tokens" in prediction:
        target_length = prediction["length"][i]
        tokens = prediction["tokens"][i][:target_length]
        sentence = self.labels_inputter.tokenizer.detokenize(tokens)
      else:
        sentence = prediction["text"][i]
      score = None
      attention = None
      if with_scores:
        score = prediction["log_probs"][i]
      if alignment_type:
        attention = prediction["alignment"][i][:target_length]
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
        ignore_weights=updated_variables)


class SequenceToSequenceInputter(inputters.ExampleInputter):
  """A custom :class:`opennmt.inputters.ExampleInputter` for sequence to
  sequence models.
  """

  def __init__(self,
               features_inputter,
               labels_inputter,
               share_parameters=False):
    super(SequenceToSequenceInputter, self).__init__(
        features_inputter, labels_inputter, share_parameters=share_parameters)
    labels_inputter.set_decoder_mode(mark_start=True, mark_end=True)
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
      labels["alignment"] = text.alignment_matrix_from_pharaoh(
          alignment,
          self.features_inputter.get_length(features, ignore_special_tokens=True),
          self.labels_inputter.get_length(labels, ignore_special_tokens=True))
    return features, labels


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
  return tf.gather(tokens, alignment, axis=1, batch_dims=1)

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

def _add_noise(tokens, lengths, params, subword_token, is_spacer=None):
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
  noiser = noise.WordNoiser(noises=noises, subword_token=subword_token, is_spacer=is_spacer)
  return noiser(tokens, lengths, keep_shape=True)
