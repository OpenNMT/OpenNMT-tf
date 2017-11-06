"""Sequence classifier."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.misc import count_lines, print_bytes


class SequenceClassifier(Model):
  """A sequence classifier."""

  def __init__(self,
               inputter,
               encoder,
               labels_vocabulary_file_key,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      labels_vocabulary_file_key: The data configuration key of the labels
        vocabulary file containing one label per line.
      name: The name of this model.
    """
    super(SequenceClassifier, self).__init__(name)

    self.inputter = inputter
    self.encoder = encoder
    self.labels_vocabulary_file_key = labels_vocabulary_file_key

  def _initialize(self, metadata):
    self.inputter.initialize(metadata)
    self.labels_vocabulary_file = metadata[self.labels_vocabulary_file_key]
    self.num_labels = count_lines(self.labels_vocabulary_file)

  def _get_serving_input_receiver(self):
    return self.inputter.get_serving_input_receiver()

  def _get_features_length(self, features):
    return self.inputter.get_length(features)

  def _get_features_builder(self, features_file):
    dataset = self.inputter.make_dataset(features_file)
    process_fn = self.inputter.process
    padded_shapes_fn = lambda: self.inputter.padded_shapes
    return dataset, process_fn, padded_shapes_fn

  def _get_labels_builder(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

    dataset = tf.data.TextLineDataset(labels_file)
    process_fn = lambda x: {
        "classes": x,
        "classes_id": labels_vocabulary.lookup(x)
    }
    padded_shapes_fn = lambda: {
        "classes": [],
        "classes_id": []
    }
    return dataset, process_fn, padded_shapes_fn

  def _build(self, features, labels, params, mode, config):
    with tf.variable_scope("encoder"):
      inputs = self.inputter.transform_data(
          features,
          mode=mode,
          log_dir=config.model_dir)

      encoder_outputs, _, _ = self.encoder.encode(
          inputs,
          sequence_length=self._get_features_length(features),
          mode=mode)

    encoding = tf.reduce_mean(encoder_outputs, axis=1)

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
          encoding,
          self.num_labels)

    if mode != tf.estimator.ModeKeys.TRAIN:
      labels_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(
          self.labels_vocabulary_file,
          vocab_size=self.num_labels)
      classes_prob = tf.nn.softmax(logits)
      classes_id = tf.argmax(classes_prob, axis=1)
      predictions = {
          "classes": labels_vocab_rev.lookup(classes_id)
      }
    else:
      predictions = None

    return logits, predictions

  def _compute_loss(self, features, labels, outputs, mode):
    return tf.losses.sparse_softmax_cross_entropy(labels["classes_id"], outputs)

  def _compute_metrics(self, features, labels, predictions):
    return {
        "accuracy": tf.metrics.accuracy(labels["classes"], predictions["classes"])
    }

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)
