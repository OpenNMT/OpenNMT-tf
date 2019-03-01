"""Sequence classifier."""

import tensorflow as tf

from opennmt import inputters
from opennmt.models.model import Model
from opennmt.utils.cell import last_encoding_from_state
from opennmt.utils.misc import print_bytes
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
        labels_inputter=ClassInputter(labels_vocabulary_file_key),
        daisy_chain_variables=daisy_chain_variables)
    self.encoder = encoder
    self.encoding = encoding.lower()
    if self.encoding not in ("average", "last"):
      raise ValueError("Invalid encoding vector: {}".format(self.encoding))

  def _call(self, features, labels, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope("encoder"):
      inputs = self.features_inputter.make_inputs(features, training=training)
      encoder_outputs, encoder_state, _ = self.encoder.encode(
          inputs,
          sequence_length=self.features_inputter.get_length(features),
          mode=mode)

    if self.encoding == "average":
      encoding = tf.reduce_mean(encoder_outputs, axis=1)
    elif self.encoding == "last":
      encoding = last_encoding_from_state(encoder_state)

    with tf.variable_scope("generator"):
      logits = tf.layers.dense(
          encoding,
          self.labels_inputter.vocabulary_size)

    if mode != tf.estimator.ModeKeys.TRAIN:
      labels_vocab_rev = self.labels_inputter.vocabulary_lookup_reverse()
      classes_prob = tf.nn.softmax(logits)
      classes_id = tf.argmax(classes_prob, axis=1)
      predictions = {
          "classes": labels_vocab_rev.lookup(classes_id)
      }
    else:
      predictions = None

    return logits, predictions

  def compute_loss(self, outputs, labels, training=True, params=None):
    if params is None:
      params = {}
    return cross_entropy_loss(
        outputs,
        labels["classes_id"],
        label_smoothing=params.get("label_smoothing", 0.0),
        training=training)

  def compute_metrics(self, predictions, labels):
    return {
        "accuracy": tf.metrics.accuracy(labels["classes"], predictions["classes"])
    }

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)


class ClassInputter(inputters.TextInputter):
  """Reading class from a text file."""

  def __init__(self, vocabulary_file_key):
    super(ClassInputter, self).__init__(
        vocabulary_file_key=vocabulary_file_key, num_oov_buckets=0)

  def make_features(self, element=None, features=None, training=None):
    return {
        "classes": element,
        "classes_id": self.vocabulary.lookup(element)
    }
