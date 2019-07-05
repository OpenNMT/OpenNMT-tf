"""Sequence classifier."""

import tensorflow as tf

from opennmt import inputters
from opennmt.models.model import Model
from opennmt.utils.misc import print_bytes
from opennmt.utils.losses import cross_entropy_loss


class SequenceClassifier(Model):
  """A sequence classifier."""

  def __init__(self, inputter, encoder):
    """Initializes a sequence classifier.

    Args:
      inputter: A :class:`opennmt.inputters.Inputter` to process the
        input data.
      encoder: A :class:`opennmt.encoders.Encoder` to encode the input.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    example_inputter = inputters.ExampleInputter(inputter, ClassInputter())
    super(SequenceClassifier, self).__init__(example_inputter)
    self.encoder = encoder

  def build(self, input_shape):
    super(SequenceClassifier, self).build(input_shape)
    self.id_to_class = self.labels_inputter.vocabulary_lookup_reverse()
    self.output_layer = tf.keras.layers.Dense(self.labels_inputter.vocabulary_size)

  def call(self, features, labels=None, step=None, training=None):
    inputs = self.features_inputter(features, training=training)
    _, state, _ = self.encoder(
        inputs,
        sequence_length=self.features_inputter.get_length(features),
        training=training)

    last_state = state[-1] if isinstance(state, (list, tuple)) else state
    encoding = last_state if not isinstance(state, (list, tuple)) else last_state[0]
    logits = self.output_layer(encoding)

    if not training:
      classes_prob = tf.nn.softmax(logits)
      classes_id = tf.argmax(classes_prob, axis=1)
      predictions = {
          "classes": self.id_to_class.lookup(classes_id),
          "classes_id": classes_id
      }
    else:
      predictions = None

    return logits, predictions

  def compute_loss(self, outputs, labels, training=True):
    return cross_entropy_loss(
        outputs,
        labels["classes_id"],
        label_smoothing=self.params.get("label_smoothing", 0.0),
        training=training)

  def get_metrics(self):
    return {"accuracy": tf.keras.metrics.Accuracy()}

  def update_metrics(self, metrics, predictions, labels):
    metrics["accuracy"].update_state(labels["classes_id"], predictions["classes_id"])

  def print_prediction(self, prediction, params=None, stream=None):
    print_bytes(prediction["classes"], stream=stream)


class ClassInputter(inputters.TextInputter):
  """Reading class from a text file."""

  def __init__(self):
    super(ClassInputter, self).__init__(num_oov_buckets=0)

  def make_features(self, element=None, features=None, training=None):
    return {
        "classes": element,
        "classes_id": self.vocabulary.lookup(element)
    }
