"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt import estimator
from opennmt import inputters
from opennmt.utils import compat
from opennmt.utils.optim import optimize_loss


@six.add_metaclass(abc.ABCMeta)
class Model(tf.keras.layers.Layer):
  """Base class for models."""

  def __init__(self,
               name,
               features_inputter=None,
               labels_inputter=None,
               daisy_chain_variables=False,
               dtype=None,
               examples_inputter=None):
    if examples_inputter is None:
      examples_inputter = inputters.ExampleInputter(features_inputter, labels_inputter)
    self.examples_inputter = examples_inputter
    if dtype is None:
      dtype = self.features_inputter.dtype
    super(Model, self).__init__(name=name, dtype=dtype)
    self.daisy_chain_variables = daisy_chain_variables

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
    with tf.variable_scope(self.name, initializer=self._initializer(params)):
      if not compat.reuse():
        self._build()  # Always rebuild unless the scope is marked for reuse.
      return self._call(features, labels, params, mode)

  def _initializer(self, params):
    """Returns the global initializer for this model.

    Args:
      params: A dictionary of hyperparameters.

    Returns:
      The initializer.
    """
    param_init = params.get("param_init")
    if param_init is not None:
      return tf.random_uniform_initializer(
          minval=-param_init, maxval=param_init, dtype=self.dtype)
    return None

  def _build(self):
    """Builds stateful layers."""
    return

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

  def input_fn(self,
               mode,
               batch_size,
               metadata=None,
               features_file=None,
               labels_file=None,
               batch_type="examples",
               batch_multiplier=1,
               bucket_width=None,
               single_pass=False,
               num_threads=None,
               sample_buffer_size=None,
               prefetch_buffer_size=None,
               maximum_features_length=None,
               maximum_labels_length=None,
               num_shards=1,
               shard_index=0):
    """Returns an input function.

    Args:
      mode: A ``tf.estimator.ModeKeys`` mode.
      batch_size: The batch size to use.
      metadata: A dictionary containing additional metadata set
        by the user. Required if ``Model.initialize()`` has not been called.
      features_file: The file containing input features.
      labels_file: The file containing output labels.
      batch_type: The training batching stragety to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier to prepare splitting accross
         replicated graph parts.
      bucket_width: The width of the length buckets to select batch candidates
        from. ``None`` to not constrain batch formation.
      single_pass: If ``True``, makes a single pass over the training data.
      num_threads: The number of elements processed in parallel.
      sample_buffer_size: The number of elements from which to sample.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
        older versions.
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length of the labels sequence.
        ``None`` to not constrain the length.
      num_shards: The number of data shards (usually the number of workers in a
        distributed setting).
      shard_index: The shard index this input pipeline should read from.

    Returns:
      A callable that returns the next element.

    See Also:
      ``tf.estimator.Estimator``.
    """
    if metadata is not None:
      self.initialize(metadata)
    return estimator.make_input_fn(
        self,
        mode,
        batch_size,
        features_file,
        labels_file=labels_file,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        bucket_width=bucket_width,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        shuffle_buffer_size=sample_buffer_size,
        single_pass=single_pass,
        num_shards=num_shards,
        shard_index=shard_index,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size,
        return_dataset=False)

  def model_fn(self, num_devices=1, eval_prediction_hooks_fn=None, devices=None, hvd=None):
    """Returns the model function.

    Args:
      num_devices: The number of devices used for training.
      eval_prediction_hooks_fn: A callable that takes the model predictions
        during evaluation and return an iterable of evaluation hooks (e.g. for
        saving predictions on disk, running external evaluators, etc.).
      devices: The list of devices used for training, if known.
      hvd: Optional Horovod object.

    See Also:
      ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
      arguments and the returned value.
    """
    return estimator.make_model_fn(
        self,
        eval_prediction_hooks_fn=eval_prediction_hooks_fn,
        num_devices=num_devices,
        devices=devices,
        hvd=hvd)

  def serving_input_fn(self, metadata=None):
    """Returns the serving input function.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user. Required if ``Model.initialize()`` has not been called.

    Returns:
      A callable that returns a ``tf.estimator.export.ServingInputReceiver``.
    """
    if metadata is not None:
      self.initialize(metadata)
    return estimator.make_serving_input_fn(self)

  def get_assets(self, metadata=None, asset_dir=None):
    """Returns additional assets used by this model.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user. Required if ``Model.initialize()`` has not been called.
      asset_dir: The directory where assets can be written.

    Returns:
      A dictionary of additional assets.
    """
    if metadata is not None:
      self.initialize(metadata)
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
