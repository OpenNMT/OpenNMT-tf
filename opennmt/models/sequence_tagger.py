"""Sequence tagger."""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from opennmt import inputters
from opennmt.models.model import Model
from opennmt.utils import misc
from opennmt.utils.losses import cross_entropy_sequence_loss


class SequenceTagger(Model):
    """A sequence tagger."""

    def __init__(self, inputter, encoder, crf_decoding=False):
        """Initializes a sequence tagger.

        Args:
          inputter: A :class:`opennmt.inputters.Inputter` to process the
            input data.
          encoder: A :class:`opennmt.encoders.Encoder` to encode the input.
          crf_decoding: If ``True``, add a CRF layer after the encoder.
        """
        example_inputter = inputters.ExampleInputter(inputter, TagsInputter())
        super().__init__(example_inputter)
        self.encoder = encoder
        self.crf_decoding = crf_decoding
        self.tagging_scheme = None
        self.transition_params = None

    def initialize(self, data_config, params=None):
        self.tagging_scheme = data_config.get("tagging_scheme")
        if self.tagging_scheme:
            self.tagging_scheme = self.tagging_scheme.lower()
        super().initialize(data_config, params=params)

    def build(self, input_shape):
        super().build(input_shape)
        num_tags = self.labels_inputter.vocabulary_size
        self.output_layer = tf.keras.layers.Dense(num_tags)
        if self.crf_decoding:
            self.transition_params = self.add_weight(
                "transition_params", [num_tags, num_tags]
            )

    def call(self, features, labels=None, training=None, step=None):
        length = self.features_inputter.get_length(features)
        inputs = self.features_inputter(features, training=training)
        outputs, _, length = self.encoder(
            inputs, sequence_length=length, training=training
        )
        logits = self.output_layer(outputs)
        if not training:
            if self.crf_decoding:
                tags_id, _ = tfa.text.crf_decode(logits, self.transition_params, length)
                tags_id = tf.cast(tags_id, tf.int64)
            else:
                tags_prob = tf.nn.softmax(logits)
                tags_id = tf.argmax(tags_prob, axis=2)
            predictions = {
                "length": tf.identity(length),
                "tags": self.labels_inputter.ids_to_tokens.lookup(tags_id),
                "tags_id": tags_id,
            }
        else:
            predictions = None
        return logits, predictions

    def compute_loss(self, outputs, labels, training=True):
        if self.crf_decoding:
            log_likelihood, _ = tfa.text.crf_log_likelihood(
                outputs,
                tf.cast(labels["tags_id"], tf.int32),
                labels["length"],
                transition_params=self.transition_params,
            )
            batch_size = tf.shape(log_likelihood)[0]
            return tf.reduce_sum(-log_likelihood) / tf.cast(
                batch_size, log_likelihood.dtype
            )
        else:
            return cross_entropy_sequence_loss(
                outputs,
                labels["tags_id"],
                sequence_length=labels["length"],
                sequence_weight=labels.get("weight"),
                label_smoothing=self.params.get("label_smoothing", 0.0),
                average_in_time=self.params.get("average_loss_in_time", False),
                training=training,
            )

    def get_metrics(self):
        metrics = {"accuracy": tf.keras.metrics.Accuracy()}
        if self.tagging_scheme in ("bioes",):
            f1 = F1()
            metrics["f1"] = f1
            metrics["precision"] = f1.precision
            metrics["recall"] = f1.recall
        return metrics

    def update_metrics(self, metrics, predictions, labels):
        weights = tf.sequence_mask(
            labels["length"], maxlen=tf.shape(labels["tags"])[1], dtype=tf.float32
        )

        metrics["accuracy"].update_state(
            labels["tags_id"], predictions["tags_id"], sample_weight=weights
        )

        if self.tagging_scheme in ("bioes",):
            flag_fn = None
            if self.tagging_scheme == "bioes":
                flag_fn = flag_bioes_tags

            gold_flags, predicted_flags = tf.numpy_function(
                flag_fn,
                [labels["tags"], predictions["tags"], labels["length"]],
                [tf.bool, tf.bool],
            )

            metrics["f1"].update_state(gold_flags, predicted_flags)

    def print_prediction(self, prediction, params=None, stream=None):
        tags = prediction["tags"][: prediction["length"]]
        sent = b" ".join(tags)
        misc.print_as_bytes(sent, stream=stream)


class TagsInputter(inputters.TextInputter):
    """Reading space-separated tags."""

    def __init__(self):
        super().__init__(num_oov_buckets=0)

    def make_features(self, element=None, features=None, training=None):
        features = super().make_features(
            element=element, features=features, training=training
        )
        return {
            "length": features["length"],
            "tags": features["tokens"],
            "tags_id": self.tokens_to_ids.lookup(features["tokens"]),
        }


class F1(tf.keras.metrics.Metric):
    """Defines a F1 metric."""

    def __init__(self, **kwargs):
        """Initializes the metric.

        Args:
          **kwargs: Base class arguments.
        """
        super().__init__(**kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    @property
    def updates(self):
        """Metric update operations."""
        return self.precision.updates + self.recall.updates

    def update_state(self, y_true, y_pred):
        """Updates the metric state."""
        self.precision.update_state(y_true, y_pred)
        self.recall.update_state(y_true, y_pred)

    def result(self):
        """Returns the metric result."""
        precision = self.precision.result()
        recall = self.recall.result()
        return (2 * precision * recall) / (recall + precision)


def flag_bioes_tags(gold, predicted, sequence_length=None):
    """Flags chunk matches for the BIOES tagging scheme.

    This function will produce the gold flags and the predicted flags. For each aligned
    gold flag ``g`` and predicted flag ``p``:

    * when ``g == p == True``, the chunk has been correctly identified (true positive).
    * when ``g == False and p == True``, the chunk has been incorrectly identified (false positive).
    * when ``g == True and p == False``, the chunk has been missed (false negative).
    * when ``g == p == False``, the chunk has been correctly ignored (true negative).

    Args:
      gold: The gold tags as a Numpy 2D string array.
      predicted: The predicted tags as a Numpy 2D string array.
      sequence_length: The length of each sequence as Numpy array.

    Returns:
      A tuple ``(gold_flags, predicted_flags)``.
    """
    gold_flags = []
    predicted_flags = []

    def _add_true_positive():
        gold_flags.append(True)
        predicted_flags.append(True)

    def _add_false_positive():
        gold_flags.append(False)
        predicted_flags.append(True)

    def _add_true_negative():
        gold_flags.append(False)
        predicted_flags.append(False)

    def _add_false_negative():
        gold_flags.append(True)
        predicted_flags.append(False)

    def _match(ref, hyp, index, length):
        if ref[index].startswith(b"B"):
            match = True
            while index < length and not ref[index].startswith(b"E"):
                if ref[index] != hyp[index]:
                    match = False
                index += 1
            match = match and index < length and ref[index] == hyp[index]
            return match, index
        return ref[index] == hyp[index], index

    for b in range(gold.shape[0]):
        length = sequence_length[b] if sequence_length is not None else gold.shape[1]

        # First pass to detect true positives and true/false negatives.
        index = 0
        while index < length:
            gold_tag = gold[b][index]
            match, index = _match(gold[b], predicted[b], index, length)
            if match:
                if gold_tag == b"O":
                    _add_true_negative()
                else:
                    _add_true_positive()
            else:
                if gold_tag != b"O":
                    _add_false_negative()
            index += 1

        # Second pass to detect false postives.
        index = 0
        while index < length:
            pred_tag = predicted[b][index]
            match, index = _match(predicted[b], gold[b], index, length)
            if not match and pred_tag != b"O":
                _add_false_positive()
            index += 1

    return np.array(gold_flags), np.array(predicted_flags)
