"""Sequence classifier."""

import tensorflow as tf

from opennmt.models.model import Model
from opennmt.utils.cell import last_encoding_from_state
from opennmt.utils.misc import count_lines, print_bytes
from opennmt.utils.losses import cross_entropy_loss


class SequenceClassifier(Model):
  """A sequence classifier."""

  def __init__(self,
               inputter,
               encoder,
               labels_vocabulary_file_key,
               encoding="average",
               daisy_chain_variables=False,
               name="seqclassifier"):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.inputter.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.encoder.Encoder` to encode the input.
      labels_vocabulary_file_key: The data configuration key of the labels
        vocabulary file containing one label per line.
      encoding: "average" or "last" (case insensitive), the encoding vector to
        extract from the encoder outputs.
      daisy_chain_variables: If ``True``, copy variables in a daisy chain
        between devices for this model. Not compatible with RNN based models.
      name: The name of this model.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(SequenceClassifier, self).__init__(
        name,
        features_inputter=inputter,
        daisy_chain_variables=daisy_chain_variables)

    self.encoder = encoder
    self.labels_vocabulary_file_key = labels_vocabulary_file_key
    self.encoding = encoding.lower()

    if self.encoding not in ("average", "last"):
      raise ValueError("Invalid encoding vector: {}".format(self.encoding))

  def _initialize(self, metadata):
    super(SequenceClassifier, self)._initialize(metadata)
    self.labels_vocabulary_file = metadata[self.labels_vocabulary_file_key]
    self.num_labels = count_lines(self.labels_vocabulary_file)

  def _get_labels_builder(self, labels_file):
    labels_vocabulary = tf.contrib.lookup.index_table_from_file(
        self.labels_vocabulary_file,
        vocab_size=self.num_labels)

    dataset = tf.data.TextLineDataset(labels_file)
    process_fn = lambda x: {
        "classes": x,
        "classes_id": labels_vocabulary.lookup(x)
    }
    return dataset, process_fn

  def _build(self, features, labels, params, mode, config=None):
    with tf.variable_scope("encoder"):
      inputs = self.features_inputter.transform_data(
          features,
          mode=mode,
          log_dir=config.model_dir if config is not None else None)

      encoder_outputs, encoder_state, _ = self.encoder.encode(
          inputs,
          sequence_length=self._get_features_length(features),
          mode=mode)

    if self.encoding == "average":
      encoding = tf.reduce_mean(encoder_outputs, axis=1)
    elif self.encoding == "last":
      encoding = last_encoding_from_state(encoder_state)

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

  def _compute_loss(self, features, labels, outputs, params, mode):
    return cross_entropy_loss(
        outputs,
        labels["classes_id"],
        label_smoothing=params.get("label_smoothing", 0.0),
        mode=mode)

  def _compute_metrics(self, features, labels, predictions):
    return {
        "accuracy": tf.metrics.accuracy(labels["classes"], predictions["classes"])
    }

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)
