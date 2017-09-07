"""Sequence classifier."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines


class SequenceClassifier(Model):

  def __init__(self,
               inputter,
               encoder,
               labels_vocabulary_file_key,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A `onmt.inputters.Inputter` to process the input data.
      encoder: A `onmt.encoders.Encoder` to encode the input.
      labels_vocabulary_file_key: The run configuration key of the labels
        vocabulary file containing one label per line.
      name: The name of this model.
    """
    super(SequenceClassifier, self).__init__(name)

    self.inputter = inputter
    self.encoder = encoder
    self.labels_vocabulary_file_key = labels_vocabulary_file_key

  def _build_features(self, features_file, resources):
    dataset = self.inputter.make_dataset(features_file, resources)
    return dataset, self.inputter.padded_shapes

  def _build_labels(self, labels_file, resources):
    self.labels_vocabulary_file = resources[self.labels_vocabulary_file_key]
    self.num_labels = count_lines(self.labels_vocabulary_file)

    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
      self.labels_vocabulary_file,
      vocab_size=self.num_labels)

    dataset = tf.contrib.data.TextLineDataset(labels_file)
    dataset = dataset.map(lambda x: labels_vocabulary.lookup(x))
    padded_shapes = []
    return dataset, padded_shapes

  def _build(self, features, labels, params, mode):
    with tf.variable_scope("encoder"):
      inputs = self.inputter.transform_data(
        features,
        mode,
        log_dir=params.get("log_dir"))

      encoder_outputs, encoder_states, encoder_sequence_length = self.encoder.encode(
        inputs,
        sequence_length=features["length"],
        mode=mode)

    encoding = tf.reduce_mean(encoder_outputs, axis=1)

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
        encoding,
        self.num_labels)

    if mode != tf.estimator.ModeKeys.PREDICT:
      loss = tf.losses.sparse_softmax_cross_entropy(
        labels,
        logits)

      return tf.estimator.EstimatorSpec(
        mode,
        loss=loss,
        train_op=self._build_train_op(loss, params))
    else:
      labels_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

      probs = tf.nn.softmax(logits)
      predictions = tf.argmax(probs, axis=1)
      predictions = labels_vocab_rev.lookup(predictions)

      return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions)
