"""Sequence tagger."""

from __future__ import print_function

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines
from opennmt.utils.losses import masked_sequence_loss


class SequenceTagger(Model):
  """A sequence tagger."""

  def __init__(self,
               inputter,
               encoder,
               labels_vocabulary_file_key,
               crf_decoding=False,
               name="seqtagger"):
    """Initializes a sequence tagger.

    Args:
      inputter: A `onmt.inputters.Inputter` to process the input data.
      encoder: A `onmt.encoders.Encoder` to encode the input.
      labels_vocabulary_file_key: The run configuration key of the labels
        vocabulary file containing one label per line.
      crf_decoding: If `True`, add a CRF layer after the encoder.
      name: The name of this model.
    """
    super(SequenceTagger, self).__init__(name)

    self.encoder = encoder
    self.inputter = inputter
    self.labels_vocabulary_file_key = labels_vocabulary_file_key
    self.crf_decoding = crf_decoding

  def _initialize(self, metadata):
    self.inputter.initialize(metadata)
    self.labels_vocabulary_file = metadata[self.labels_vocabulary_file_key]
    self.num_labels = count_lines(self.labels_vocabulary_file)

  def _get_serving_input_receiver(self):
    return self.inputter.get_serving_input_receiver()

  def _get_features_length(self, features):
    return self.inputter.get_length(features)

  def _get_labels_length(self, labels):
    return None

  def _get_features_builder(self, features_file):
    dataset = self.inputter.make_dataset(features_file)
    process_fn = self.inputter.process
    padded_shapes_fn = lambda: self.inputter.padded_shapes
    return dataset, process_fn, padded_shapes_fn

  def _get_labels_builder(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

    dataset = tf.contrib.data.TextLineDataset(labels_file)
    process_fn = lambda x: labels_vocabulary.lookup(tf.string_split([x]).values)
    padded_shapes_fn = lambda: [None]
    return dataset, process_fn, padded_shapes_fn

  def _build(self, features, labels, params, mode):
    length = self._get_features_length(features)

    with tf.variable_scope("encoder"):
      inputs = self.inputter.transform_data(
          features,
          mode,
          log_dir=params.get("log_dir"))

      encoder_outputs, _, encoder_sequence_length = self.encoder.encode(
          inputs,
          sequence_length=length,
          mode=mode)

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
          encoder_outputs,
          self.num_labels)

    if mode != tf.estimator.ModeKeys.PREDICT:
      if self.crf_decoding:
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
            logits,
            tf.cast(labels, tf.int32),
            length)
        loss = tf.reduce_mean(-log_likelihood)
      else:
        loss = masked_sequence_loss(
            logits,
            labels,
            length)

      return tf.estimator.EstimatorSpec(
          mode,
          loss=loss,
          train_op=self._build_train_op(loss, params))
    else:
      if self.crf_decoding:
        transition_params = tf.get_variable(
            "transitions", shape=[self.num_labels, self.num_labels])
        labels, _ = tf.contrib.crf.crf_decode(
            logits,
            transition_params,
            encoder_sequence_length)
        labels = tf.cast(labels, tf.int64)
      else:
        probs = tf.nn.softmax(logits)
        labels = tf.argmax(probs, axis=2)

      labels_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.labels_vocabulary_file,
          vocab_size=self.num_labels)

      predictions = {}
      predictions["length"] = encoder_sequence_length
      predictions["labels"] = labels_vocab_rev.lookup(labels)

      export_outputs = {
          "predictions": tf.estimator.export.PredictOutput(predictions)
      }

      return tf.estimator.EstimatorSpec(
          mode,
          predictions=predictions,
          export_outputs=export_outputs)

  def print_prediction(self, prediction, params=None):
    labels = prediction["labels"][:prediction["length"]]
    sent = b" ".join(labels)
    print(sent.decode("utf-8"))
