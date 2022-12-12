"""Base class for models."""

import abc
import contextlib

import tensorflow as tf

from opennmt import optimizers, schedules
from opennmt.utils import exporters, losses, misc


class Model(tf.keras.layers.Layer):
    """Base class for models."""

    def __init__(self, examples_inputter):
        super().__init__()
        self.examples_inputter = examples_inputter
        self.params = {}

    @property
    def unsupervised(self):
        """Unsupervised model."""
        return self.labels_inputter is None

    @property
    def features_inputter(self):
        """The inputter producing features."""
        return getattr(
            self.examples_inputter, "features_inputter", self.examples_inputter
        )

    @property
    def labels_inputter(self):
        """The inputter producing labels."""
        return getattr(self.examples_inputter, "labels_inputter", None)

    def __repr__(self):
        """Returns a description of the model and its submodules."""
        return misc.describe_layer(self, name="model")

    def auto_config(self, num_replicas=1):
        """Returns automatic configuration values specific to this model.

        Args:
          num_replicas: The number of synchronous model replicas used for the
            training.

        Returns:
          A partial training configuration.
        """
        _ = num_replicas
        return {}

    def initialize(self, data_config, params=None):
        """Initializes the model from the data configuration.

        Args:
          data_config: A dictionary containing the data configuration set
            by the user (e.g. vocabularies, tokenization, pretrained embeddings,
            etc.).
          params: A dictionary of hyperparameters.
        """
        if params is None:
            params = {}
        self.params.update(params)
        dropout = self.params.get("dropout")
        if dropout is not None:
            misc.set_dropout(self, dropout)
        self.examples_inputter.initialize(data_config)

    def build(self, input_shape):
        freeze_layers = self.params.get("freeze_layers")
        if freeze_layers:
            if not isinstance(freeze_layers, list):
                freeze_layers = [freeze_layers]
            for layer_path in freeze_layers:
                layer = misc.index_structure(self, layer_path)
                layer.trainable = False
                misc.set_dropout(layer, 0)  # Disable dropout in frozen layers.
        self.examples_inputter.build(input_shape)
        self.built = True

    def split_features_labels(self, batch):
        """Splits a batch from the dataset into features and labels."""
        return batch if not self.unsupervised else (batch, batch)

    def __call__(self, features, labels=None, training=None, step=None):
        """Runs the model.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.
          training: If ``True``, run in training mode.
          step: The current training step.

        Returns:
          A tuple containing,

          - The model outputs (usually unscaled probabilities).
          - The model predictions.
        """
        outputs, predictions = super().__call__(
            features,
            labels=labels,
            training=training,
            step=step,
        )

        # Include the example index vector in the outputs.
        index = features.get("index") if isinstance(features, dict) else None
        if index is not None:
            if isinstance(outputs, dict):
                outputs["index"] = index
            if isinstance(predictions, dict):
                predictions["index"] = index

        return outputs, predictions

    @abc.abstractmethod
    def call(self, features, labels=None, training=None, step=None):
        """Runs the model.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.
          training: If ``True``, run in training mode.
          step: The current training step.

        Returns:
          A tuple containing,

          - The model outputs (usually unscaled probabilities).
          - The model predictions.
        """
        raise NotImplementedError()

    def infer(self, features):
        """Runs inference on :obj:`features`.

        This is a small convenience wrapper around
        :meth:`opennmt.models.Model.call`.

        Args:
          features: A nested structure of features ``tf.Tensor``.

        Returns:
          The model predictions.
        """
        _, predictions = self(features)
        return predictions

    def evaluate(self, features, labels):
        """Evaluates :obj:`features` predictions against `labels`.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.

        Returns:
          A tuple with the loss and the model predictions.
        """
        outputs, predictions = self(features, labels=labels)
        loss = self.compute_loss(outputs, labels, training=False)
        return loss, predictions

    def score(self, features, labels):
        """Scores labels.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.

        Returns:
          The score results.
        """
        raise NotImplementedError("This model does not define a score function")

    def train(self, features, labels, optimizer, loss_scale=None):
        """Computes and applies the gradients for a batch of examples.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.
          optimizer: The optimizer instance
            (``tf.keras.mixed_precision.LossScaleOptimizer`` is supported).
          loss_scale: An optional loss scaling factor.

        Returns:
          The loss.
        """
        loss, gradients = self.compute_gradients(
            features,
            labels,
            optimizer,
            loss_scale=loss_scale,
        )
        optimizer.apply_gradients(list(zip(gradients, self.trainable_weights)))
        return loss

    def compute_gradients(self, features, labels, optimizer, loss_scale=None):
        """Computes the gradients for a batch of examples.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.
          optimizer: The optimizer instance
            (``tf.keras.mixed_precision.LossScaleOptimizer`` is supported).
          loss_scale: An optional loss scaling factor.

        Returns:
          A tuple containing,

          - The loss.
          - The gradients.
        """

        def _compute_loss():
            train_loss, report_loss = self.compute_training_loss(
                features,
                labels,
                step=optimizer.iterations,
            )
            if loss_scale is not None:
                train_loss /= loss_scale
                report_loss /= loss_scale
            return train_loss, report_loss

        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                train_loss, report_loss = _compute_loss()
                if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                    train_loss = optimizer.get_scaled_loss(train_loss)
            gradients = tape.gradient(train_loss, self.trainable_weights)
            if isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
                gradients = optimizer.get_unscaled_gradients(gradients)

        else:
            train_loss, report_loss = _compute_loss()
            # LossScaleOptimizer.get_gradients takes care of loss scaling.
            gradients = optimizer.get_gradients(train_loss, self.trainable_weights)

        return report_loss, gradients

    def compute_training_loss(self, features, labels, step=None):
        """Computes the training loss for a batch of examples.

        Args:
          features: A nested structure of features ``tf.Tensor``.
          labels: A nested structure of labels ``tf.Tensor``.
          step: The current training step.

        Returns:
          A tuple containing,

          - The loss to optimize.
          - The loss to report.
        """
        outputs, _ = self(features, labels, training=True, step=step)
        loss = self.compute_loss(outputs, labels, training=True)
        if isinstance(loss, tuple):
            train_loss = loss[0] / loss[1]
            report_loss = loss[0] / loss[2] if len(loss) > 2 else train_loss
        else:
            train_loss, report_loss = loss, loss
        train_loss = self.regularize_loss(train_loss, variables=self.trainable_weights)
        return train_loss, report_loss

    @abc.abstractmethod
    def compute_loss(self, outputs, labels, training=True):
        """Computes the loss.

        Args:
          outputs: The model outputs (usually unscaled probabilities).
          labels: The dict of labels ``tf.Tensor``.
          training: If ``True``, compute the loss for training.

        Returns:
          The loss or a tuple ``(numerator, train_denominator, stats_denominator)``
          to use a different normalization for training compared to reporting (e.g.
          batch-normalized for training vs. token-normalized for reporting).
        """
        raise NotImplementedError()

    def regularize_loss(self, loss, variables=None):
        """Regularizes the loss.

        Args:
          loss: The loss.
          variables: List of variables.

        Returns:
          The regularized loss.
        """
        if variables is None:
            variables = self.trainable_variables
        regularization = self.params.get("regularization")
        if regularization is not None:
            loss += losses.regularization_penalty(
                regularization["type"], regularization["scale"], variables
            )
        return loss

    def get_metrics(self):
        """Returns the metrics for this model.

        Returns:
          A dictionary of ``tf.keras.metrics.Metric`` metrics.
        """
        return None

    def update_metrics(self, metrics, predictions, labels):
        """Computes additional metrics on the predictions.

        Args:
          metrics: A dictionary of metrics to update.
          predictions: The model predictions.
          labels: The dict of labels ``tf.Tensor``.
        """
        return

    def get_optimizer(self):
        """Returns the optimizer for this model.

        Returns:
          A ``tf.keras.optimizers.legacy.Optimizer`` instance or ``None`` if no optimizer
          is configured.
        """
        params = self.params
        optimizer_name = params.get("optimizer")
        if optimizer_name is None:
            return None
        schedule_type = params.get("decay_type")
        if schedule_type is None:
            learning_rate = tf.constant(params["learning_rate"], dtype=tf.float32)
        else:
            schedule_params = params.get("decay_params", {})
            learning_rate = schedules.make_learning_rate_schedule(
                params.get("learning_rate"),
                schedule_type,
                schedule_params=schedule_params,
                schedule_step_duration=params.get("decay_step_duration", 1),
                start_step=params.get("start_decay_steps", 0),
                minimum_learning_rate=params.get("minimum_learning_rate", 0),
            )
        optimizer_params = params.get("optimizer_params")
        if optimizer_params is None:
            optimizer_params = {}
        optimizer = optimizers.make_optimizer(
            optimizer_name, learning_rate, **optimizer_params
        )
        return optimizer

    def serve_function(self):
        """Returns a function for serving this model.

        Returns:
          A ``tf.function``.
        """
        # Set name attribute of the input TensorSpec.
        input_signature = {
            name: tf.TensorSpec.from_spec(spec, name=name)
            for name, spec in self.features_inputter.input_signature().items()
        }

        @tf.function(input_signature=(input_signature,))
        def _run(features):
            features = self.features_inputter.make_features(features=features.copy())
            _, predictions = self(features)
            return predictions

        return _run

    @property
    def tflite_mode(self):
        """Returns ``True`` if the model is being traced for TensorFlow Lite."""
        return getattr(self, "_tflite_mode", False)

    @contextlib.contextmanager
    def enable_tflite_mode(self):
        """Enable TensorFlow Lite mode for this model."""
        layers = [self] + list(self.submodules)
        for layer in layers:
            setattr(layer, "_tflite_mode", True)
        yield
        for layer in layers:
            delattr(layer, "_tflite_mode")

    def tflite_function(self):
        """Returns the inference function that should be used for TensorFlow Lite.

        Returns:
          A ``tf.function``.
        """
        with self.enable_tflite_mode():
            return self.serve_function()

    def export(self, export_dir, exporter=None):
        """Exports the model for serving.

        Args:
          export_dir: The output directory.
          exporter: A :class:`opennmt.utils.Exporter` instance. Defaults to
            :class:`opennmt.utils.SavedModelExporter`.
        """
        if exporter is None:
            exporter = exporters.SavedModelExporter()
        exporter.export(self, export_dir)

    def create_variables(self, optimizer=None):
        """Creates the model variables by running it once.

        Args:
          optimizer: If set, also create the optimizer variables.
        """
        # Create input features from the input signatures. We remove the leading
        # batch dimension as sometimes assumed by make_features methods and set
        # unspecified dimensions to 1.
        features = tf.nest.map_structure(
            lambda spec: tf.fill(
                [dim or 1 for dim in spec.shape.as_list()[1:]],
                tf.constant("a" if spec.dtype is tf.string else 1, dtype=spec.dtype),
            ),
            self.examples_inputter.input_signature(),
        )
        features = self.examples_inputter.make_features(features=features)

        # Add the batch dimension back before calling the model.
        features = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
        features, labels = self.split_features_labels(features)

        _ = self(features, labels=labels, training=True, step=0)

        if optimizer is not None:
            optimizer._create_all_weights(self.trainable_variables)

    def transfer_weights(
        self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None
    ):
        """Transfers weights (and optionally optimizer slots) from this model to
        another.

        This default implementation assumes that :obj:`self` and :obj:`new_model`
        have exactly the same variables. Subclasses can override this method to
        transfer weights to another model type or architecture. For example,
        :class:`opennmt.models.SequenceToSequence` can transfer weights to a model
        with a different vocabulary.

        All model and optimizer variables are expected to be initialized.

        Args:
          new_model: The new model to transfer weights to.
          new_optimizer: The new optimizer.
          optimizer: The optimizer used for the current model.
          ignore_weights: Optional list of weights to not transfer.
        """
        if type(self) is not type(new_model):
            raise ValueError(
                "Transferring weights to another model type is not supported"
            )
        if ignore_weights is None:
            ignore_weights = set()
        ignore_weights_ref = set(weight.ref() for weight in ignore_weights)
        weights = self.weights
        new_weights = new_model.weights
        for weight, new_weight in zip(weights, new_weights):
            if new_weight.ref() not in ignore_weights_ref:
                new_weight.assign(weight)
                if new_optimizer is not None and optimizer is not None:
                    for slot_name in new_optimizer.get_slot_names():
                        if slot_name not in optimizer.get_slot_names():
                            continue
                        new_slot = new_optimizer.get_slot(new_weight, slot_name)
                        slot = optimizer.get_slot(weight, slot_name)
                        new_slot.assign(slot)

    def map_v1_weights(self, weights):
        """Maps current weights to V1 weights.

        Args:
          weights: A nested dictionary following the scope names used in V1. The
            leaves are tuples with the variable value and optionally the optimizer
            slots.

        Returns:
          A list of tuples associating variables and their V1 equivalent.
        """
        raise NotImplementedError("This model can not restore V1 checkpoints")

    def export_assets(self, asset_dir):
        """Exports additional assets used by this model.

        Args:
          asset_dir: The directory where assets can be written.

        Returns:
          A dictionary of additional assets.
        """
        return self.examples_inputter.export_assets(asset_dir)

    def visualize(self, log_dir):
        """Setups model visualization (e.g. word embedding projections).

        Args:
          log_dir: The log directory.
        """
        self.features_inputter.visualize(self, log_dir)
        if not self.unsupervised:
            self.labels_inputter.visualize(self, log_dir)

    def format_prediction(self, prediction, params=None):
        """Formats the model prediction for file saving.

        Args:
          prediction: The model prediction (same structure as the second output of
            :meth:`opennmt.models.Model.call`).
          params: (optional) Dictionary of formatting parameters.

        Returns:
          A string or list of strings.
        """
        return str(prediction)

    def format_score(self, score, params=None, stream=None):
        """Formats the score result for file saving.

        Args:
          score: The score result (same structure as the output of
            :meth:`opennmt.models.Model.score`).
          params: (optional) Dictionary of formatting parameters.
        """
        return str(score)

    def print_prediction(self, prediction, params=None, stream=None):
        """Prints the model prediction.

        Args:
          prediction: The model prediction (same structure as the second output of
            :meth:`opennmt.models.Model.call`).
          params: (optional) Dictionary of formatting parameters.
          stream: (optional) The stream to print to.
        """
        _write_lines(self.format_prediction(prediction, params=params), stream)

    def print_score(self, score, params=None, stream=None):
        """Prints the score result.

        Args:
          score: The score result (same structure as the output of
            :meth:`opennmt.models.Model.score`).
          params: (optional) Dictionary of formatting parameters.
          stream: (optional) The stream to print to.
        """
        _write_lines(self.format_score(score, params=params), stream)


def _write_lines(lines, stream):
    if not isinstance(lines, list):
        lines = [lines]
    for line in lines:
        misc.print_as_bytes(line, stream=stream)


class SequenceGenerator(Model):
    """Base class for models generating sequences."""

    @property
    def decoder_inputter(self):
        """The inputter used on the decoder side."""
        return self.labels_inputter if not self.unsupervised else self.examples_inputter

    def score(self, features, labels):
        outputs, _ = self(features, labels=labels)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels["ids_out"], outputs["logits"]
        )
        weights = tf.sequence_mask(labels["length"], dtype=cross_entropy.dtype)
        masked_cross_entropy = cross_entropy * weights
        scores = tf.reduce_sum(masked_cross_entropy, axis=1)
        results = {
            "cross_entropy": cross_entropy,
            "score": scores,
            "tokens": labels["tokens"],
            "length": self.decoder_inputter.get_length(
                labels, ignore_special_tokens=True
            ),
        }
        for key_to_forward in ("attention", "index"):
            value = outputs.get(key_to_forward)
            if value is not None:
                results[key_to_forward] = value
        return results

    def format_score(self, score, params=None):
        if params is None:
            params = {}
        length = score["length"]
        tokens = score["tokens"][:length]
        sentence = " ".join(token.decode("utf-8") for token in tokens)
        token_level_scores = None
        attention = None
        if params.get("with_token_level"):
            token_level_scores = score["cross_entropy"][:length]
        if "attention" in score:
            attention = score["attention"][:length]
        alignment_type = params.get("with_alignments")
        return misc.format_translation_output(
            sentence,
            score=score["score"],
            token_level_scores=token_level_scores,
            attention=attention,
            alignment_type=alignment_type,
        )
