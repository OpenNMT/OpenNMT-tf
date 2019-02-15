"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt import inputters
from opennmt.utils import hooks
from opennmt.utils.optim import optimize_loss
from opennmt.utils.parallel import GraphDispatcher


@six.add_metaclass(abc.ABCMeta)
class Model(object):
  """Base class for models."""

  def __init__(self,
               name,
               features_inputter=None,
               labels_inputter=None,
               daisy_chain_variables=False,
               dtype=None,
               examples_inputter=None):
    self.name = name
    self.examples_inputter = examples_inputter
    if self.examples_inputter is None:
      self.examples_inputter = inputters.ExampleInputter(features_inputter, labels_inputter)
    self.features_inputter = self.examples_inputter.features_inputter
    self.labels_inputter = self.examples_inputter.labels_inputter
    self.daisy_chain_variables = daisy_chain_variables
    if dtype is None and self.features_inputter is not None:
      self.dtype = self.features_inputter.dtype
    else:
      self.dtype = dtype or tf.float32

  def auto_config(self, num_devices=1):
    """Returns automatic configuration values specific to this model.

    Args:
      num_devices: The number of devices used for the training.

    Returns:
      A partial training configuration.
    """
    _ = num_devices
    return {}

  def __call__(self, features, labels, params, mode, config=None):
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
    return self._build(features, labels, params, mode, config=config)

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
    dispatcher = GraphDispatcher(
        num_devices=num_devices,
        daisy_chain_variables=self.daisy_chain_variables,
        devices=devices)

    def _loss_op(features, labels, params, mode, config):
      """Single callable to compute the loss."""
      logits, _ = self._build(features, labels, params, mode, config=config)
      return self._compute_loss(features, labels, logits, params, mode)

    def _normalize_loss(num, den=None):
      """Normalizes the loss."""
      if isinstance(num, list):  # Sharded mode.
        if den is not None:
          assert isinstance(den, list)
          return tf.add_n(num) / tf.add_n(den)
        else:
          return tf.reduce_mean(num)
      elif den is not None:
        return num / den
      else:
        return num

    def _extract_loss(loss):
      """Extracts and summarizes the loss."""
      if not isinstance(loss, tuple):
        actual_loss = _normalize_loss(loss)
        tboard_loss = actual_loss
      else:
        actual_loss = _normalize_loss(loss[0], den=loss[1])
        tboard_loss = _normalize_loss(loss[0], den=loss[2]) if len(loss) > 2 else actual_loss
      tf.summary.scalar("loss", tboard_loss)
      return actual_loss

    def _model_fn(features, labels, params, mode, config):
      """model_fn implementation."""
      if mode == tf.estimator.ModeKeys.TRAIN:
        features_shards = dispatcher.shard(features)
        labels_shards = dispatcher.shard(labels)

        with tf.variable_scope(self.name, initializer=self._initializer(params)):
          losses_shards = dispatcher(
              _loss_op, features_shards, labels_shards, params, mode, config)

        loss = _extract_loss(losses_shards)
        train_op, extra_variables = optimize_loss(
            loss, params, mixed_precision=(self.dtype == tf.float16), hvd=hvd)

        training_hooks = []
        if extra_variables:
          training_hooks.append(hooks.VariablesInitializerHook(extra_variables))
        if config is not None:
          self.examples_inputter.visualize(config.model_dir)
          features_length = self.features_inputter.get_length(features)
          labels_length = self.labels_inputter.get_length(labels)
          num_words = {}
          if features_length is not None:
            num_words["source"] = tf.reduce_sum(features_length)
          if labels_length is not None:
            num_words["target"] = tf.reduce_sum(labels_length)
          training_hooks.append(hooks.LogWordsPerSecondHook(
              num_words,
              every_n_steps=config.save_summary_steps,
              output_dir=config.model_dir))
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            training_hooks=training_hooks)
      elif mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope(self.name):
          logits, predictions = self._build(features, labels, params, mode, config=config)
          loss = self._compute_loss(features, labels, logits, params, mode)

        loss = _extract_loss(loss)
        eval_metric_ops = self._compute_metrics(features, labels, predictions)  # pylint: disable=assignment-from-none
        evaluation_hooks = []
        if predictions is not None and eval_prediction_hooks_fn is not None:
          evaluation_hooks.extend(eval_prediction_hooks_fn(predictions))
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops,
            evaluation_hooks=evaluation_hooks)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope(self.name):
          _, predictions = self._build(features, labels, params, mode, config=config)

        # Forward example index for reordering predictions.
        if "index" in features:
          predictions["index"] = features["index"]

        export_outputs = {}
        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
            tf.estimator.export.PredictOutput(predictions))

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs)
      else:
        raise RuntimeError("Invalid mode")

    return _model_fn

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

  @abc.abstractmethod
  def _build(self, features, labels, params, mode, config=None):
    """Creates the graph.

    Returns:
      outputs: The model outputs (usually unscaled probabilities).
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.PREDICT``.
      predictions: The model predictions.
        Optional if :obj:`mode` is ``tf.estimator.ModeKeys.TRAIN``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _compute_loss(self, features, labels, outputs, params, mode):
    """Computes the loss.

    Args:
      features: The dict of features ``tf.Tensor``.
      labels: The dict of labels ``tf.Tensor``.
      output: The model outputs (usually unscaled probabilities).
      params: A dictionary of hyperparameters.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      The loss or a tuple containing the computed loss and the loss to display.
    """
    raise NotImplementedError()

  def _compute_metrics(self, features, labels, predictions):  # pylint: disable=unused-argument
    """Computes additional metrics on the predictions.

    Args:
      features: The dict of features ``tf.Tensor``.
      labels: The dict of labels ``tf.Tensor``.
      predictions: The model predictions.

    Returns:
      A dict of metric results (tuple ``(metric_tensor, update_op)``) keyed by
      name.
    """
    return None

  def _initialize(self, metadata, asset_dir=None):
    """Runs model specific initialization (e.g. vocabularies loading).

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
      asset_dir: The directory where assets can be written. If ``None``, no
        assets are returned.

    Returns:
      A dictionary containing additional assets used by the model.
    """
    return self.examples_inputter.initialize(metadata, asset_dir=asset_dir)

  def input_fn(self,
               mode,
               batch_size,
               metadata,
               features_file,
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
        by the user.
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
    batch_size_multiple = 1
    if batch_type == "tokens" and self.dtype == tf.float16:
      batch_size_multiple = 8

    def _fn():
      self._initialize(metadata)

      if mode == tf.estimator.ModeKeys.PREDICT:
        dataset = self.examples_inputter.make_inference_dataset(
            features_file,
            batch_size,
            bucket_width=bucket_width,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)
      elif mode == tf.estimator.ModeKeys.EVAL:
        dataset = self.examples_inputter.make_evaluation_dataset(
            features_file,
            labels_file,
            batch_size,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)
      elif mode == tf.estimator.ModeKeys.TRAIN:
        dataset = self.examples_inputter.make_training_dataset(
            features_file,
            labels_file,
            batch_size,
            batch_type=batch_type,
            batch_multiplier=batch_multiplier,
            batch_size_multiple=batch_size_multiple,
            shuffle_buffer_size=sample_buffer_size,
            bucket_width=bucket_width,
            maximum_features_length=maximum_features_length,
            maximum_labels_length=maximum_labels_length,
            single_pass=single_pass,
            num_shards=num_shards,
            shard_index=shard_index,
            num_threads=num_threads,
            prefetch_buffer_size=prefetch_buffer_size)

      iterator = dataset.make_initializable_iterator()
      # Add the initializer to a standard collection for it to be initialized.
      tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
      return iterator.get_next()

    return _fn

  def serving_input_fn(self, metadata):
    """Returns the serving input function.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.

    Returns:
      A callable that returns a ``tf.estimator.export.ServingInputReceiver``.
    """
    if self.features_inputter is None:
      raise NotImplementedError()

    def _fn():
      self._initialize(metadata)
      # This is a hack for SequenceRecordInputter that currently infers the input
      # depth from the data files.
      # TODO: This method should not require the training data.
      if  "train_features_file" in metadata:
        _ = self.features_inputter.make_dataset(metadata["train_features_file"])
      return self.features_inputter.get_serving_input_receiver()

    return _fn

  def get_assets(self, metadata, asset_dir):
    """Returns additional assets used by this model.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
      asset_dir: The directory where assets can be written.

    Returns:
      A dictionary of additional assets.
    """
    assets = self._initialize(metadata, asset_dir=asset_dir)
    tf.reset_default_graph()
    return assets

  def print_prediction(self, prediction, params=None, stream=None):
    """Prints the model prediction.

    Args:
      prediction: The evaluated prediction.
      params: (optional) Dictionary of formatting parameters.
      stream: (optional) The stream to print to.
    """
    _ = params
    print(prediction, file=stream)
