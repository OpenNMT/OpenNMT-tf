"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt.utils.optim import optimize_loss


@six.add_metaclass(abc.ABCMeta)
class Model(tf.keras.layers.Layer):
  """Base class for models."""

  def __init__(self,
               examples_inputter,
               dtype=None,
               name=None):
    self.examples_inputter = examples_inputter
    if dtype is None:
      dtype = self.features_inputter.dtype
    super(Model, self).__init__(name=name, dtype=dtype)

  @property
  def unsupervised(self):
    """Unsupervised model."""
    return not hasattr(self.examples_inputter, "labels_inputter")

  @property
  def features_inputter(self):
    """The inputter producing features."""
    return getattr(self.examples_inputter, "features_inputter", self.examples_inputter)

  @property
  def labels_inputter(self):
    """The inputter producing labels."""
    return self.examples_inputter.labels_inputter

  def auto_config(self, num_devices=1):
    """Returns automatic configuration values specific to this model.

    Args:
      num_devices: The number of devices used for the training.

    Returns:
      A partial training configuration.
    """
    _ = num_devices
    return {}

  def initialize(self, metadata):
    """Initializes the model from the data configuration.

    Args:
      metadata: A dictionary containing additional data configuration set
        by the user (e.g. vocabularies, tokenization, pretrained embeddings,
        etc.).
    """
    self.examples_inputter.initialize(metadata)

  def __call__(self, features, labels, params, mode, config=None):  # pylint: disable=arguments-differ
    """Calls the model function.

    Returns:
      outputs: The model outputs (usually unscaled probabilities).
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
      predictions: The model predictions.
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.

    See Also:
      ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
      the arguments of this function.
    """
    if not self.built:
      self._build()
    return self._call(features, labels, params, mode)

  def _build(self):
    """Builds stateful layers."""
    self.examples_inputter.build()
    self.built = True

  @abc.abstractmethod
  def _call(self, features, labels, params, mode):
    """Creates the graph.

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

  def optimize_loss(self, loss, params=None, hvd=None):
    """Returns the loss optimization op.

    Args:
      loss: The loss to optimize.
      params: A dictionary of hyperparameters.
      hvd: Optional Horovod object.

    Returns:
      The training op and optionally a list of extra variables to initialize.
    """
    if params is None:
      params = {}
    mixed_precision = self.dtype == tf.float16
    return optimize_loss(loss, params, mixed_precision=mixed_precision, hvd=hvd)

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
