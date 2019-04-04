"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt.utils import optim


@six.add_metaclass(abc.ABCMeta)
class Model(tf.keras.Model):
  """Base class for models."""

  def __init__(self, examples_inputter):
    super(Model, self).__init__()
    self.examples_inputter = examples_inputter

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

  def initialize(self, data_config):
    """Initializes the model from the data configuration.

    Args:
      data_config: A dictionary containing the data configuration set
        by the user (e.g. vocabularies, tokenization, pretrained embeddings,
        etc.).
    """
    self.examples_inputter.initialize(data_config)

  def build(self, input_shape):
    self.examples_inputter.build(input_shape)
    self.built = True

  @abc.abstractmethod
  def call(self, features, labels, params, mode):
    """Runs the model.

    Args:
      features: A nested structure of features ``tf.Tensor``.
      labels: A nested structure of labels ``tf.Tensor``.
      params: A dictionary of hyperparameters.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      outputs: The model outputs (usually unscaled probabilities).
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
      predictions: The model predictions.
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def compute_loss(self, outputs, labels, training=True, params=None):
    """Computes the loss.

    Args:
      outputs: The model outputs (usually unscaled probabilities).
      labels: The dict of labels ``tf.Tensor``.
      training: Compute training loss.
      params: A dictionary of hyperparameters.

    Returns:
      The loss or a tuple containing the computed loss and the loss to display.
    """
    raise NotImplementedError()

  def compute_metrics(self, predictions, labels):  # pylint: disable=unused-argument
    """Computes additional metrics on the predictions.

    Args:
      predictions: The model predictions.
      labels: The dict of labels ``tf.Tensor``.

    Returns:
      A dict of metrics. See the ``eval_metric_ops`` field of
      ``tf.estimator.EstimatorSpec``.
    """
    return None

  def get_optimizer(self, params=None):
    """Returns the optimizer for this model.

    Args:
      params: A dictionary of hyperparameters.

    Returns:
      A ``tf.optimizers.Optimizer`` instance.
    """
    if params is None:
      params = {}
    learning_rate = tf.constant(params["learning_rate"], dtype=tf.float32)
    if params.get("decay_type") is not None:
      decay_params = params.get("decay_params", {})
      learning_rate = optim.make_learning_rate_schedule(
          learning_rate,
          params["decay_type"],
          decay_params=decay_params,
          decay_step_duration=params.get("decay_step_duration", 1),
          start_decay_step=params.get("start_decay_steps", 0),
          minimum_learning_rate=params.get("minimum_learning_rate", 0))
    optimizer = optim.make_optimizer(
        params["optimizer"],
        learning_rate,
        **params.get("optimizer_params", {}))
    return optimizer

  def compute_gradients(self, loss, optimizer, params=None):
    """Computes the gradients.

    Args:
      loss: The loss.
      optimizer: The ``tf.optimizers.Optimizer`` instance.
      params: A dictionary of hyperparameters.

    Returns:
      The list of gradients.
    """
    if params is None:
      params = {}
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

  def apply_gradients(self, gradients, optimizer, params=None, step=None):
    """Applies the gradients.

    Args:
      gradients: The list of gradients to apply.
      optimizer: The ``tf.optimizers.Optimizer`` instance.
      params: A dictionary of hyperparameters.
      step: An optional step counter to increment when the parameters are
        updated.

    Returns:
      An operation that applies the gradients and optionally a list of internal
      variables to initialize.
    """
    if params is None:
      params = {}
    return optim.delayed_update(
        optimizer,
        list(zip(gradients, self.trainable_variables)),
        accum_count=params.get("gradients_accum", 1),
        global_step=step)

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
