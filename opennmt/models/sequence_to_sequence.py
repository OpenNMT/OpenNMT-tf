"""Standard sequence-to-sequence model."""

import tensorflow as tf

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

  data = inputter.set_data_field(
      data,
      "ids_out",
      tf.concat([ids, eos], axis=0),
      padded_shape=[None])
  data = inputter.set_data_field(
      data,
      "ids",
      tf.concat([bos, ids], axis=0),
      padded_shape=[None])

  # Increment length accordingly.
  inputter.set_data_field(data, "length", length + 1)

  return data


class SequenceToSequence(Model):
  """A sequence to sequence model."""

  def __init__(self,
               source_inputter,
               target_inputter,
               encoder,
               decoder,
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
      name: The name of this model.

    Raises:
      TypeError: if :obj:`target_inputter` is not a
        :class:`opennmt.inputters.text_inputter.WordEmbedder`.
    """
    super(SequenceToSequence, self).__init__(name)

    if not isinstance(target_inputter, inputters.WordEmbedder):
      raise TypeError("Target inputter must be a WordEmbedder")

    self.encoder = encoder
    self.decoder = decoder

    self.source_inputter = source_inputter
    self.target_inputter = target_inputter
    self.target_inputter.add_process_hooks([shift_target_sequence])

  def _initialize(self, metadata):
    self.source_inputter.initialize(metadata)
    self.target_inputter.initialize(metadata)

  def _get_serving_input_receiver(self):
    return self.source_inputter.get_serving_input_receiver()

  def _get_features_length(self, features):
    return self.source_inputter.get_length(features)

  def _get_labels_length(self, labels):
    return self.target_inputter.get_length(labels)

  def _get_features_builder(self, features_file):
    dataset = self.source_inputter.make_dataset(features_file)
    process_fn = self.source_inputter.process
    padded_shapes_fn = lambda: self.source_inputter.padded_shapes
    return dataset, process_fn, padded_shapes_fn

  def _get_labels_builder(self, labels_file):
    dataset = self.target_inputter.make_dataset(labels_file)
    process_fn = self.target_inputter.process
    padded_shapes_fn = lambda: self.target_inputter.padded_shapes
    return dataset, process_fn, padded_shapes_fn

  def _scoped_target_embedding_fn(self, mode, scope):
    def _target_embedding_fn(ids):
      try:
        with tf.variable_scope(scope):
          return self.target_inputter.transform(ids, mode=mode)
      except ValueError:
        with tf.variable_scope(scope, reuse=True):
          return self.target_inputter.transform(ids, mode=mode)
    return _target_embedding_fn

  def _build(self, features, labels, params, mode, config):
    features_length = self._get_features_length(features)

    with tf.variable_scope("encoder"):
      source_inputs = self.source_inputter.transform_data(
          features,
          mode=mode,
          log_dir=config.model_dir)
      encoder_outputs, encoder_state, encoder_sequence_length = self.encoder.encode(
          source_inputs,
          sequence_length=features_length,
          mode=mode)

    target_vocab_size = self.target_inputter.vocabulary_size

    with tf.variable_scope("decoder") as decoder_scope:
      if labels is not None:
        sampling_probability = get_sampling_probability(
            tf.train.get_or_create_global_step(),
            read_probability=params.get("scheduled_sampling_read_probability"),
            schedule_type=params.get("scheduled_sampling_type"),
            k=params.get("scheduled_sampling_k"))

        target_inputs = self.target_inputter.transform_data(
            labels,
            mode=mode,
            log_dir=config.model_dir)
        logits, _, _ = self.decoder.decode(
            target_inputs,
            self._get_labels_length(labels),
            target_vocab_size,
            initial_state=encoder_state,
            sampling_probability=sampling_probability,
            embedding=self._scoped_target_embedding_fn(mode, decoder_scope),
            mode=mode,
            memory=encoder_outputs,
            memory_sequence_length=encoder_sequence_length)
      else:
        logits = None

    if mode != tf.estimator.ModeKeys.TRAIN:
      with tf.variable_scope(decoder_scope, reuse=labels is not None) as decoder_scope:
        batch_size = tf.shape(encoder_sequence_length)[0]
        beam_width = params.get("beam_width", 1)
        maximum_iterations = params.get("maximum_iterations", 250)
        start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
        end_token = constants.END_OF_SENTENCE_ID

        if beam_width <= 1:
          sampled_ids, _, sampled_length, log_probs = self.decoder.dynamic_decode(
              self._scoped_target_embedding_fn(mode, decoder_scope),
              start_tokens,
              end_token,
              target_vocab_size,
              initial_state=encoder_state,
              maximum_iterations=maximum_iterations,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length)
        else:
          length_penalty = params.get("length_penalty", 0)
          sampled_ids, _, sampled_length, log_probs = self.decoder.dynamic_decode_and_search(
              self._scoped_target_embedding_fn(mode, decoder_scope),
              start_tokens,
              end_token,
              target_vocab_size,
              initial_state=encoder_state,
              beam_width=beam_width,
              length_penalty=length_penalty,
              maximum_iterations=maximum_iterations,
              mode=mode,
              memory=encoder_outputs,
              memory_sequence_length=encoder_sequence_length)

      target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.target_inputter.vocabulary_file,
          vocab_size=target_vocab_size - self.target_inputter.num_oov_buckets,
          default_value=constants.UNKNOWN_TOKEN)

      predictions = {
          "tokens": target_vocab_rev.lookup(tf.cast(sampled_ids, tf.int64)),
          "length": sampled_length,
          "log_probs": log_probs
      }
    else:
      predictions = None

    return logits, predictions

  def _compute_loss(self, features, labels, outputs, params, mode):
    return cross_entropy_sequence_loss(
        outputs,
        labels["ids_out"],
        self._get_labels_length(labels),
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        mode=mode)

  def print_prediction(self, prediction, params=None, stream=None):
    n_best = params and params.get("n_best")
    n_best = n_best or 1

    if n_best > len(prediction["tokens"]):
      raise ValueError("n_best cannot be greater than beam_width")

    for i in range(n_best):
      tokens = prediction["tokens"][i][:prediction["length"][i] - 1] # Ignore </s>.
      sentence = self.target_inputter.tokenizer.detokenize(tokens)
      print_bytes(tf.compat.as_bytes(sentence), stream=stream)
