"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt import optimizers
from opennmt import schedules
from opennmt.utils import misc


@six.add_metaclass(abc.ABCMeta)
class Model(tf.keras.layers.Layer):
  """Base class for models."""

  def __init__(self, examples_inputter):
    super(Model, self).__init__()
    self.examples_inputter = examples_inputter
    self.params = {}
    self._frozen_layers = None

  @property
  def unsupervised(self):
    """Unsupervised model."""
    return self.labels_inputter is None

  @property
  def features_inputter(self):
    """The inputter producing features."""
    return getattr(self.examples_inputter, "features_inputter", self.examples_inputter)

  @property
  def labels_inputter(self):
    """The inputter producing labels."""
    return getattr(self.examples_inputter, "labels_inputter", None)

  @property
  def trainable_weights(self):
    if self._frozen_layers is None:
      self._frozen_layers = []
      freeze_layers = self.params.get("freeze_layers")
      if freeze_layers:
        if not isinstance(freeze_layers, list):
          freeze_layers = [freeze_layers]
        for layer_path in freeze_layers:
          layer = misc.index_structure(self, layer_path)
          layer.trainable = False
          self._frozen_layers.append(layer)
    return super(Model, self).trainable_weights

  def auto_config(self, num_replicas=1):
    """Returns automatic configuration values specific to this model.

    Args:
      num_replicas: The number of concurrent model replicas used for the
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
    self.examples_inputter.initialize(data_config)

  def build(self, input_shape):
    self.examples_inputter.build(input_shape)
    self.built = True

  @abc.abstractmethod
  def call(self, features, labels=None, training=None, step=None):  # pylint: disable=arguments-differ
    """Runs the model.

    Args:
      features: A nested structure of features ``tf.Tensor``.
      labels: A nested structure of labels ``tf.Tensor``.
      training: Run in training mode.
      step: The current training step.

    Returns:
      A tuple containing,

      - The model outputs (usually unscaled probabilities).
      - The model predictions.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_loss(self, outputs, labels, training=True):
    """Computes the loss.

    Args:
      outputs: The model outputs (usually unscaled probabilities).
      labels: The dict of labels ``tf.Tensor``.
      training: Compute training loss.

    Returns:
      The loss or a tuple ``(numerator, train_denominator, stats_denominator)``
      to use a different normalization for training compared to reporting (e.g.
      batch-normalized for training vs. token-normalized for reporting).
    """
    raise NotImplementedError()

  def get_metrics(self):
    """Returns the metrics for this model.

    Returns:
      A dictionary of ``tf.keras.metrics.Metric`` metrics.
    """
    return None

  def update_metrics(self, metrics, predictions, labels):  # pylint: disable=unused-argument
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
      A ``tf.keras.optimizers.Optimizer`` instance.
    """
    params = self.params
    learning_rate = tf.constant(params["learning_rate"], dtype=tf.float32)
    if params.get("decay_type") is not None:
      schedule_params = params.get("decay_params", {})
      learning_rate = schedules.make_learning_rate_schedule(
          learning_rate,
          params["decay_type"],
          schedule_params=schedule_params,
          start_step=params.get("start_decay_steps", 0),
          minimum_learning_rate=params.get("minimum_learning_rate", 0))
    optimizer = optimizers.make_optimizer(
        params["optimizer"],
        learning_rate,
        **params.get("optimizer_params", {}))
    return optimizer

  def compute_gradients(self, loss, optimizer, variables=None):
    """Computes the gradients.

    Args:
      loss: The loss.
      optimizer: The ``tf.keras.optimizers.Optimizer`` instance.
      variables: List of variables.

    Returns:
      The list of gradients.
    """
    params = self.params
    if variables is None:
      variables = self.trainable_variables
    regularization = params.get("regularization")
    if regularization is not None:
      loss += optim.regularization_penalty(
          regularization["type"], regularization["scale"], variables)
    gradients = optimizer.get_gradients(loss, variables)
    clip_gradients = params.get("clip_gradients")
    if clip_gradients is not None:
      gradients, _ = tf.clip_by_global_norm(gradients, float(clip_gradients))
    return gradients

  def serve_function(self):
    """Returns a function for serving this model.

    Returns:
      A ``tf.function``.
    """
    # Set name attribute of the input TensorSpec.
    input_signature = {
        name:tf.TensorSpec.from_spec(spec, name=name)
        for name, spec in six.iteritems(self.features_inputter.input_signature())}

    @tf.function(input_signature=(input_signature,))
    def _run(features):
      features = self.features_inputter.make_features(features=features.copy())
      _, predictions = self(features)
      return predictions

    return _run

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once.

    Args:
      optimizer: If set, also create the optimizer variables.
    """
    if self.built:
      return

    # Create input features from the input signatures. We remove the leading
    # batch dimension as sometimes assumed by make_features methods and set
    # unspecified dimensions to 1.
    features = tf.nest.map_structure(
        lambda spec: tf.zeros(
            [dim or 1 for dim in spec.shape.as_list()[1:]],
            dtype=spec.dtype),
        self.examples_inputter.input_signature())
    features = self.examples_inputter.make_features(features=list(features))

    # Add the batch dimension back before calling the model.
    features, labels = tf.nest.map_structure(lambda x: tf.expand_dims(x, 0), features)
    _ = self(features, labels=labels, training=True, step=0)

    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()  # pylint: disable=protected-access
      optimizer._create_slots(self.trainable_variables)  # pylint: disable=protected-access

  def transfer_weights(self, new_model, new_optimizer=None, optimizer=None, ignore_weights=None):
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
      ignore_weights: Optional set of weights to not transfer.
    """
    if type(self) is not type(new_model):
      raise ValueError("Transferring weights to another model type is not supported")
    if ignore_weights is None:
      ignore_weights = set()
    weights = self.trainable_weights
    new_weights = new_model.weights
    if (new_optimizer is not None and optimizer is not None
        and len(new_optimizer.weights) == len(optimizer.weights)):
      weights += optimizer.weights
      new_weights += new_optimizer.weights
    for weight, new_weight in zip(weights, new_weights):
      if new_weight not in ignore_weights:
        new_weight.assign(weight)

  def map_v1_weights(self, weights):
    """Maps current weights to V1 weights.

    Args:
      weights: A nested dictionary following the scope names used in V1. The
        leaves are tuples with the variable value and optionally the optimizer
        slots.

    Returns:
      A dictionary mapping variables to their V1 equivalent.
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
      prediction: The evaluated prediction.
      params: (optional) Dictionary of formatting parameters.
      stream: (optional) The stream to print to.
    """
    _ = params
    print(prediction, file=stream)
