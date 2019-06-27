"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt import optimizers


@six.add_metaclass(abc.ABCMeta)
class Model(tf.keras.Model):
  """Base class for models."""

  def __init__(self, examples_inputter):
    super(Model, self).__init__()
    self.examples_inputter = examples_inputter
    self.params = {}

  @property
  def dtype(self):
    """The model dtype."""
    return self.examples_inputter.dtype

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
  def trainable_variables(self):
    trainable_variables = super(Model, self).trainable_variables
    freeze_variables = self.params.get("freeze_variables")
    if freeze_variables:
      trainable_variables = _get_trainable_variables(trainable_variables, freeze_variables)
    return trainable_variables

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
  def call(self, features, labels=None, step=None, mode=tf.estimator.ModeKeys.PREDICT):
    """Runs the model.

    Args:
      features: A nested structure of features ``tf.Tensor``.
      labels: A nested structure of labels ``tf.Tensor``.
      step: The current training step.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      outputs: The model outputs (usually unscaled probabilities).
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
      predictions: The model predictions.
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.
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
      The loss or a tuple containing the computed loss and the loss to display.
    """
    raise NotImplementedError()

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
    pass

  def get_optimizer(self):
    """Returns the optimizer for this model.

    Returns:
      A ``tf.keras.optimizers.Optimizer`` instance.
    """
    params = self.params
    learning_rate = tf.constant(params["learning_rate"], dtype=tf.float32)
    if params.get("decay_type") is not None:
      schedule_params = params.get("decay_params", {})
      learning_rate = optimizers.schedules.make_learning_rate_schedule(
          learning_rate,
          params["decay_type"],
          schedule_params=schedule_params,
          schedule_step_duration=params.get("decay_step_duration", 1),
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

  def create_variables(self, optimizer=None):
    """Creates the model variables by running it once."""
    if self.built:
      return

    @tf.function(input_signature=(self.features_inputter.input_signature(),))
    def _run(features):
      features = self.features_inputter.make_features(features=features.copy())
      self(features)

    _run.get_concrete_function()
    if optimizer is not None:
      _ = optimizer.iterations
      optimizer._create_hypers()
      optimizer._create_slots(self.trainable_variables)

  def get_assets(self, asset_dir):
    """Returns additional assets used by this model.

    Args:
      asset_dir: The directory where assets can be written.

    Returns:
      A dictionary of additional assets.
    """
    return self.examples_inputter.export_assets(asset_dir)

  def print_prediction(self, prediction, params=None, stream=None):
    """Prints the model prediction.

    Args:
      prediction: The evaluated prediction.
      params: (optional) Dictionary of formatting parameters.
      stream: (optional) The stream to print to.
    """
    _ = params
    print(prediction, file=stream)


def _print_variables(variables, name=None):
  if name is not None:
    tf.get_logger().info("%s variables:", name.capitalize())
  for variable in variables:
    tf.get_logger().info(" * %s", variable.name)

def _get_trainable_variables(variables, freeze_variables):
  if not isinstance(freeze_variables, list):
    freeze_variables = [freeze_variables]
  regexs = list(map(re.compile, freeze_variables))
  frozen_variables = []
  trainable_variables = []
  for variable in variables:
    if any(regex.match(variable.name) for regex in regexs):
      frozen_variables.append(variable)
    else:
      trainable_variables.append(variable)
  _print_variables(frozen_variables, name="frozen")
  _print_variables(trainable_variables, name="trainable")
  return trainable_variables
