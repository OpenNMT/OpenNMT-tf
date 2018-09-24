"""Base class for models."""

from __future__ import print_function

import abc
import six

import tensorflow as tf

from opennmt.utils import data
from opennmt.utils.optim import optimize
from opennmt.utils.hooks import add_counter
from opennmt.utils.misc import add_dict_to_collection, item_or_tuple
from opennmt.utils.parallel import GraphDispatcher


@six.add_metaclass(abc.ABCMeta)
class Model(object):
  """Base class for models."""

  def __init__(self,
               name,
               features_inputter=None,
               labels_inputter=None,
               daisy_chain_variables=False,
               dtype=None):
    self.name = name
    self.features_inputter = features_inputter
    self.labels_inputter = labels_inputter
    self.daisy_chain_variables = daisy_chain_variables
    if dtype is None and self.features_inputter is not None:
      self.dtype = features_inputter.dtype
    else:
      self.dtype = dtype or tf.float32

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

  def model_fn(self, num_devices=1):
    """Returns the model function.

    Args:
      num_devices: The number of devices used for training.

    See Also:
      ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
      arguments and the returned value.
    """
    dispatcher = GraphDispatcher(
        num_devices, daisy_chain_variables=self.daisy_chain_variables)

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
        self._register_word_counters(features, labels)

        features_shards = dispatcher.shard(features)
        labels_shards = dispatcher.shard(labels)

        with tf.variable_scope(self.name, initializer=self._initializer(params)):
          losses_shards = dispatcher(
              _loss_op, features_shards, labels_shards, params, mode, config)

        loss = _extract_loss(losses_shards)
        train_op = optimize(loss, params)
        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op)
      elif mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope(self.name):
          logits, predictions = self._build(features, labels, params, mode, config=config)
          loss = self._compute_loss(features, labels, logits, params, mode)

        loss = _extract_loss(loss)
        eval_metric_ops = self._compute_metrics(features, labels, predictions)
        if predictions is not None:
          # Register predictions in a collection so that hooks can easily fetch them.
          add_dict_to_collection("predictions", predictions)

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=eval_metric_ops)
      elif mode == tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope(self.name):
          _, predictions = self._build(features, labels, params, mode, config=config)

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

  def _register_word_counters(self, features, labels):
    """Creates word counters for sequences (if any) of :obj:`features` and
    :obj:`labels`.
    """
    features_length = self._get_features_length(features)
    labels_length = self._get_labels_length(labels)

    with tf.variable_scope("words_per_sec"):
      if features_length is not None:
        add_counter("features", tf.reduce_sum(features_length))
      if labels_length is not None:
        add_counter("labels", tf.reduce_sum(labels_length))

  def _initialize(self, metadata):
    """Runs model specific initialization (e.g. vocabularies loading).

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    if self.features_inputter is not None:
      self.features_inputter.initialize(metadata)
    if self.labels_inputter is not None:
      self.labels_inputter.initialize(metadata)

  def _get_serving_input_receiver(self):
    """Returns an input receiver for serving this model.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    return self.features_inputter.get_serving_input_receiver()

  def _get_features_length(self, features):
    """Returns the features length.

    Args:
      features: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor`` or list of ``tf.Tensor``, or ``None`` if
      length is undefined.
    """
    if self.features_inputter is None:
      return None
    return self.features_inputter.get_length(features)

  def _get_labels_length(self, labels):
    """Returns the labels length.

    Args:
      labels: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor``  or ``None`` if length is undefined.
    """
    if self.labels_inputter is None:
      return None
    return self.labels_inputter.get_length(labels)

  def _get_dataset_size(self, features_file):
    """Returns the size of the dataset.

    Args:
      features_file: The file of features.

    Returns:
      The total size.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    return self.features_inputter.get_dataset_size(features_file)

  def _get_features_builder(self, features_file):
    """Returns the recipe to build features.

    Args:
      features_file: The file of features.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    if self.features_inputter is None:
      raise NotImplementedError()
    dataset = self.features_inputter.make_dataset(features_file)
    process_fn = self.features_inputter.process
    return dataset, process_fn

  def _get_labels_builder(self, labels_file):
    """Returns the recipe to build labels.

    Args:
      labels_file: The file of labels.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    if self.labels_inputter is None:
      raise NotImplementedError()
    dataset = self.labels_inputter.make_dataset(labels_file)
    process_fn = self.labels_inputter.process
    return dataset, process_fn

  def _augment_parallel_dataset(self, dataset, process_fn, mode=None):
    """Augments a parallel dataset.

    Args:
      dataset: A parallel dataset.
      process_fn: The current dataset processing function.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn)``.
    """
    _ = mode
    return dataset, process_fn

  def _input_fn_impl(self,
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
                     maximum_labels_length=None):
    """See ``input_fn``."""
    self._initialize(metadata)

    feat_dataset, feat_process_fn = self._get_features_builder(features_file)

    if labels_file is None:
      dataset = feat_dataset
      # Parallel inputs must be catched in a single tuple and not considered as multiple arguments.
      process_fn = lambda *arg: feat_process_fn(item_or_tuple(arg))
    else:
      labels_dataset, labels_process_fn = self._get_labels_builder(labels_file)

      dataset = tf.data.Dataset.zip((feat_dataset, labels_dataset))
      process_fn = lambda features, labels: (
          feat_process_fn(features), labels_process_fn(labels))
      dataset, process_fn = self._augment_parallel_dataset(dataset, process_fn, mode=mode)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = data.training_pipeline(
          dataset,
          batch_size,
          batch_type=batch_type,
          batch_multiplier=batch_multiplier,
          bucket_width=bucket_width,
          single_pass=single_pass,
          process_fn=process_fn,
          num_threads=num_threads,
          shuffle_buffer_size=sample_buffer_size,
          prefetch_buffer_size=prefetch_buffer_size,
          dataset_size=self._get_dataset_size(features_file),
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length,
          features_length_fn=self._get_features_length,
          labels_length_fn=self._get_labels_length)
    else:
      dataset = data.inference_pipeline(
          dataset,
          batch_size,
          process_fn=process_fn,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)

    iterator = dataset.make_initializable_iterator()

    # Add the initializer to a standard collection for it to be initialized.
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()

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
               maximum_labels_length=None):
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

    Returns:
      A callable that returns the next element.

    Raises:
      ValueError: if :obj:`labels_file` is not set when in training or
        evaluation mode.

    See Also:
      ``tf.estimator.Estimator``.
    """
    if mode != tf.estimator.ModeKeys.PREDICT and labels_file is None:
      raise ValueError("Labels file is required for training and evaluation")

    return lambda: self._input_fn_impl(
        mode,
        batch_size,
        metadata,
        features_file,
        labels_file=labels_file,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        bucket_width=bucket_width,
        single_pass=single_pass,
        num_threads=num_threads,
        sample_buffer_size=sample_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length)

  def _serving_input_fn_impl(self, metadata):
    """See ``serving_input_fn``."""
    self._initialize(metadata)
    return self._get_serving_input_receiver()

  def serving_input_fn(self, metadata):
    """Returns the serving input function.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.

    Returns:
      A callable that returns a ``tf.estimator.export.ServingInputReceiver``.
    """
    return lambda: self._serving_input_fn_impl(metadata)

  def print_prediction(self, prediction, params=None, stream=None):
    """Prints the model prediction.

    Args:
      prediction: The evaluated prediction.
      params: (optional) Dictionary of formatting parameters.
      stream: (optional) The stream to print to.
    """
    _ = params
    print(prediction, file=stream)
