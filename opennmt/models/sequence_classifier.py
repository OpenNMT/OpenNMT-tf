"""Sequence classifier."""

import tensorflow as tf

from opennmt import inputters
from opennmt.models.model import Model
from opennmt.utils import misc
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
        super().__init__(example_inputter)
        self.encoder = encoder

    def build(self, input_shape):
        super().build(input_shape)
        self.output_layer = tf.keras.layers.Dense(self.labels_inputter.vocabulary_size)

    def call(self, features, labels=None, training=None, step=None):
        inputs = self.features_inputter(features, training=training)
        outputs, state, outputs_length = self.encoder(
            inputs,
            sequence_length=self.features_inputter.get_length(features),
            training=training,
        )

        if state is None:
            if outputs_length is not None:
                outputs = tf.RaggedTensor.from_tensor(outputs, lengths=outputs_length)
            encoding = tf.reduce_mean(outputs, axis=1)
        else:
            last_state = state[-1] if isinstance(state, (list, tuple)) else state
            encoding = (
                last_state if not isinstance(state, (list, tuple)) else last_state[0]
            )
        logits = self.output_layer(encoding)

        if not training:
            classes_prob = tf.nn.softmax(logits)
            classes_id = tf.argmax(classes_prob, axis=1)
            predictions = {
                "classes": self.labels_inputter.ids_to_tokens.lookup(classes_id),
                "classes_id": classes_id,
            }
        else:
            predictions = None

        return logits, predictions

    def compute_loss(self, outputs, labels, training=True):
        return cross_entropy_loss(
            outputs,
            labels["classes_id"],
            weight=labels.get("weight"),
            label_smoothing=self.params.get("label_smoothing", 0.0),
            training=training,
        )

    def get_metrics(self):
        return {"accuracy": tf.keras.metrics.Accuracy()}

    def update_metrics(self, metrics, predictions, labels):
        metrics["accuracy"].update_state(
            labels["classes_id"], predictions["classes_id"]
        )

    def print_prediction(self, prediction, params=None, stream=None):
        misc.print_as_bytes(prediction["classes"], stream=stream)


class ClassInputter(inputters.TextInputter):
    """Reading class from a text file."""

    def __init__(self):
        super().__init__(num_oov_buckets=0)

    def make_features(self, element=None, features=None, training=None):
        if features is None:
            features = {}
        if "classes" not in features:
            features["classes"] = element
        features["classes_id"] = self.tokens_to_ids.lookup(features["classes"])
        return features

    def input_signature(self):
        return {"classes": tf.TensorSpec([None], tf.string)}
