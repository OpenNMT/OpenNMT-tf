"""Base class for models."""

import abc

import tensorflow as tf

from opennmt import optimizers
from opennmt import schedules
from opennmt.utils import exporters
from opennmt.utils import losses
from opennmt.utils import misc


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

    @property
    def ctranslate2_spec(self):
        """The equivalent CTranslate2 model specification."""
        return None

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
        _forward_example_index(features, predictions)
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
        _forward_example_index(features, predictions)
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
          A ``tf.keras.optimizers.Optimizer`` instance or ``None`` if no optimizer
          is configured.
        """
        params = self.params
        optimizer_name = params.get("optimizer")
        if optimizer_name is None:
            return None
        learning_rate = tf.constant(params["learning_rate"], dtype=tf.float32)
        if params.get("decay_type") is not None:
            schedule_params = params.get("decay_params", {})
            learning_rate = schedules.make_learning_rate_schedule(
                learning_rate,
                params["decay_type"],
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
            if isinstance(features, (list, tuple)):
                # Special case for unsupervised inputters that always return a
                # tuple (features, labels).
                features = features[0]
            _, predictions = self(features)
            return predictions

        return _run

    def tflite_function(self):
        """Returns the inference function that should be used for TensorFlow Lite.

        Returns:
          A ``tf.function``.
        """
        raise NotImplementedError(
            "This model does not define a function for TensorFlow Lite"
        )

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
        features, labels = tf.nest.map_structure(
            lambda x: tf.expand_dims(x, 0), features
        )
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

    def print_prediction(self, prediction, params=None, stream=None):
        """Prints the model prediction.

        Args:
          prediction: The model prediction.
          params: (optional) Dictionary of formatting parameters.
          stream: (optional) The stream to print to.
        """
        _ = params
        print(prediction, file=stream)

    def print_score(self, score, params=None, stream=None):
        """Prints the score result.

        Args:
          score: The score result (output of :meth:`opennmt.models.Model.score`).
          params: (optional) Dictionary of formatting parameters.
          stream: (optional) The stream to print to.
        """
        _ = params
        print(score, file=stream)


class SequenceGenerator(Model):
    """Base class for models generating sequences."""

    @property
    def decoder_inputter(self):
        """The inputter used on the decoder side."""
        return self.labels_inputter if not self.unsupervised else self.features_inputter

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
        if "attention" in outputs:
            results["attention"] = outputs["attention"]
        _forward_example_index(features, results)
        return results

    def print_score(self, score, params=None, stream=None):
        if params is None:
            params = {}
        length = score["length"]
        tokens = score["tokens"][:length]
        sentence = self.decoder_inputter.tokenizer.detokenize(tokens)
        token_level_scores = None
        attention = None
        if params.get("with_token_level"):
            token_level_scores = score["cross_entropy"][:length]
        if "attention" in score:
            attention = score["attention"][:length]
        alignment_type = params.get("with_alignments")
        sentence = misc.format_translation_output(
            sentence,
            score=score["score"],
            token_level_scores=token_level_scores,
            attention=attention,
            alignment_type=alignment_type,
        )
        misc.print_as_bytes(sentence, stream=stream)


def _forward_example_index(features, output):
    if isinstance(features, dict) and isinstance(output, dict) and "index" in features:
        output["index"] = features["index"]
