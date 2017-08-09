"""Standard sequence-to-sequence model."""

import tensorflow as tf

import opennmt.constants as constants

from opennmt.models.model import Model
from opennmt.utils.losses import masked_sequence_loss


class SequenceToSequence(Model):

  def __init__(self,
               source_embedder,
               target_embedder,
               encoder,
               decoder,
               name="seq2seq"):
    super(SequenceToSequence, self).__init__(name)

    self.encoder = encoder
    self.decoder = decoder

    self.source_embedder = source_embedder
    self.target_embedder = target_embedder

    self.source_embedder.set_name("source")
    self.target_embedder.set_name("target")

  def set_filters(self,
                  maximum_source_length,
                  maximum_target_length):
    self.maximum_source_length = maximum_source_length
    self.maximum_target_length = maximum_target_length

  def _get_size(self, features, labels):
    return self.source_embedder.get_data_field(features, "length")

  def _get_maximum_size(self):
    return getattr(self, "maximum_source_length", None)

  def _filter_example(self, features, labels):
    """Filters examples with invalid length."""
    cond = tf.logical_and(
      tf.greater(self.source_embedder.get_data_field(features, "length"), 0),
      tf.greater(self.target_embedder.get_data_field(labels, "length"), 0))

    if hasattr(self, "maximum_source_length"):
      cond = tf.logical_and(
        cond,
        tf.less_equal(self.source_embedder.get_data_field(features, "length"),
                      self.maximum_source_length))

    if hasattr(self, "maximum_target_length"):
      cond = tf.logical_and(
        cond,
        tf.less_equal(self.target_embedder.get_data_field(labels, "length"),
                      self.maximum_target_length + 1)) # "+ 1" because <s> was already added.

    return cond

  def _shift_target(self, labels):
    """Generate shifted target sequences with <s> and </s>."""
    bos = tf.cast(tf.constant([constants.START_OF_SENTENCE_ID]), tf.int64)
    eos = tf.cast(tf.constant([constants.END_OF_SENTENCE_ID]), tf.int64)

    ids = self.target_embedder.get_data_field(labels, "ids")
    length = self.target_embedder.get_data_field(labels, "length")

    labels = self.target_embedder.set_data_field(
      labels,
      "ids_out",
      tf.concat([ids, eos], axis=0),
      padded_shape=[None])
    labels = self.target_embedder.set_data_field(
      labels,
      "ids",
      tf.concat([bos, ids], axis=0),
      padded_shape=[None])

    # Increment length accordingly.
    self.target_embedder.set_data_field(labels, "length", length + 1)

    return labels

  def _build_dataset(self, mode, batch_size, features_file, labels_file=None):
    source_file = features_file
    target_file = labels_file

    source_dataset = tf.contrib.data.TextLineDataset(source_file)
    self.source_embedder.init()
    source_dataset = source_dataset.map(lambda x: self.source_embedder.process(x))

    if target_file is None:
      dataset = source_dataset
      padded_shapes = self.source_embedder.padded_shapes
    else:
      target_dataset = tf.contrib.data.TextLineDataset(target_file)

      self.target_embedder.init()
      target_dataset = target_dataset.map(lambda x: self.target_embedder.process(x))
      target_dataset = target_dataset.map(lambda x: self._shift_target(x))
      dataset = tf.contrib.data.Dataset.zip((source_dataset, target_dataset))
      padded_shapes = (self.source_embedder.padded_shapes, self.target_embedder.padded_shapes)

    return dataset, padded_shapes

  def _build(self, features, labels, params, mode):
    batch_size = tf.shape(self.source_embedder.get_data_field(features, "length"))[0]

    with tf.variable_scope("encoder"):
      source_inputs = self.source_embedder.embed_from_data(features, mode)
      self.source_embedder.visualize(params["log_dir"])

      encoder_outputs, encoder_states, encoder_sequence_length = self.encoder.encode(
        source_inputs,
        sequence_length=self.source_embedder.get_data_field(features, "length"),
        mode=mode)

    with tf.variable_scope("decoder") as decoder_scope:
      if mode != tf.estimator.ModeKeys.PREDICT:
        target_inputs = self.target_embedder.embed_from_data(labels, mode)
        self.target_embedder.visualize(params["log_dir"])

        decoder_outputs, _, decoded_length = self.decoder.decode(
          target_inputs,
          self.target_embedder.get_data_field(labels, "length"),
          self.target_embedder.vocabulary_size,
          encoder_states,
          mode=mode,
          memory=encoder_outputs,
          memory_sequence_length=encoder_sequence_length)
      else:
        decoder_outputs, _, decoded_length = self.decoder.dynamic_decode_and_search(
          lambda x: self.target_embedder.embed(x, mode, scope=decoder_scope, reuse_next=True),
          tf.fill([batch_size], constants.START_OF_SENTENCE_ID),
          constants.END_OF_SENTENCE_ID,
          self.target_embedder.vocabulary_size,
          encoder_states,
          beam_width=params.get("beam_width") or 5,
          length_penalty=params.get("length_penalty") or 0.0,
          maximum_iterations=params.get("maximum_iterations") or 250,
          mode=mode,
          memory=encoder_outputs,
          memory_sequence_length=encoder_sequence_length)

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = masked_sequence_loss(
        decoder_outputs,
        self.target_embedder.get_data_field(labels, "ids_out"),
        self.target_embedder.get_data_field(labels, "length"))

      return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=self._build_train_op(loss, params))
    else:
      target_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        self.target_embedder.vocabulary_file,
        vocab_size=self.target_embedder.vocabulary_size - self.target_embedder.num_oov_buckets,
        default_value=constants.UNKNOWN_TOKEN)
      predicted_ids = tf.transpose(decoder_outputs, perm=[0, 2, 1])
      predictions = {}
      predictions["tokens"] = target_vocab_rev.lookup(tf.cast(predicted_ids, tf.int64))
      predictions["length"] = decoded_length

      return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions)

  def format_prediction(self, prediction, params=None):
    n_best = params and params.get("n_best")
    n_best = n_best or 1

    all_preds = []
    for i in range(n_best):
      tokens = prediction["tokens"][i][:prediction["length"][i] - 1] # Ignore </s>.
      sentence = b' '.join(tokens)
      sentence = sentence.decode('utf-8')
      all_preds.append(sentence)

    return all_preds
