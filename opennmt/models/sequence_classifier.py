"""Sequence classifier."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines


class SequenceClassifier(Model):

  def __init__(self,
               embedder,
               encoder,
               labels_vocabulary_file,
               name="seqclassifier"):
    super(SequenceClassifier, self).__init__(name)

    self.embedder = embedder
    self.encoder = encoder
    self.labels_vocabulary_file = labels_vocabulary_file
    self.num_labels = count_lines(labels_vocabulary_file)

  def _features_length(self, features):
    return self.embedder.get_data_field(features, "length")

  def _build_features(self, features_file):
    dataset = self.embedder.make_dataset(features_file)
    return dataset, self.embedder.padded_shapes

  def _build_labels(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
      self.labels_vocabulary_file,
      vocab_size=self.num_labels)

    dataset = tf.contrib.data.TextLineDataset(labels_file)
    dataset = dataset.map(lambda x: labels_vocabulary.lookup(x))
    padded_shapes = []
    return dataset, padded_shapes

  def _build(self, features, labels, params, mode):
    with tf.variable_scope("encoder"):
      inputs = self.embedder.embed_from_data(
        features,
        mode,
        log_dir=params.get("log_dir"))

      encoder_outputs, encoder_states, encoder_sequence_length = self.encoder.encode(
        inputs,
        sequence_length=self.embedder.get_data_field(features, "length"),
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
