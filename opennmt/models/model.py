"""Base class for models."""

from __future__ import print_function

import time
import abc
import six

import tensorflow as tf

from tensorflow.python.client import device_lib

from opennmt.utils import decay
from opennmt.utils.misc import add_dict_to_collection, item_or_tuple, scalar_summary
from opennmt.utils.replicate_model_fn import replicate_model_fn, TowerOptimizer


def learning_rate_decay_fn(decay_type,
                           decay_rate,
                           decay_steps,
                           decay_step_duration=1,
                           staircase=True,
                           start_decay_steps=0,
                           minimum_learning_rate=0):
  """Returns the learning rate decay functions.

  Args:
    decay_type: The type of decay. A function from ``tf.train`` or
     :mod:`opennmt.utils.decay` as a string.
    decay_rate: The decay rate to apply.
    decay_steps: The decay steps as described in the decay type function.
    decay_step_duration: The number of training steps that make 1 decay step.
    staircase: If ``True``, learning rate is decayed in a staircase fashion.
    start_decay_steps: Start decay after this many steps.
    minimum_learning_rate: Do not decay past this learning rate value.

  Returns:
    A function with signature
    ``(learning_rate, global_step) -> decayed_learning_rate``.

  Raises:
    ValueError: if :obj:`decay_type` can not be resolved.
  """
  def _decay_fn(learning_rate, global_step):
    decay_op_name = None

    if decay_op_name is None:
      decay_op_name = getattr(tf.train, decay_type, None)
    if decay_op_name is None:
      decay_op_name = getattr(decay, decay_type, None)
    if decay_op_name is None:
      raise ValueError("Unknown decay function: {}".format(decay_type))

    # Map the training step to a decay step.
    step = tf.maximum(global_step - start_decay_steps, 0)
    step = tf.div(step, decay_step_duration)

    decayed_learning_rate = decay_op_name(
        learning_rate,
        step,
        decay_steps,
        decay_rate,
        staircase=staircase)
    decayed_learning_rate = tf.maximum(decayed_learning_rate, minimum_learning_rate)

    return decayed_learning_rate

  return _decay_fn

def get_optimizer_class(classname):
  """Returns the optimizer class.

  Args:
    classname: The name of the optimizer class in ``tf.train`` or
      ``tf.contrib.opt`` as a string.

  Returns:
    A class inheriting from ``tf.train.Optimizer``.

  Raises:
    ValueError: if :obj:`classname` can not be resolved.
  """
  optimizer_class = None

  if optimizer_class is None:
    optimizer_class = getattr(tf.train, classname, None)
  if optimizer_class is None:
    optimizer_class = getattr(tf.contrib.opt, classname, None)
  if optimizer_class is None:
    raise ValueError("Unknown optimizer class: {}".format(classname))

  return optimizer_class

def get_optimizer(global_step, params, replicated_model=False):
  """Returns an optimizer.

  Args:
    global_step: The training step.
    params: A dictionary with the user parameters.
    replicated_model: ``True`` if the model is replicated accross devices.

  Returns:
    A ``tf.train.Optimizer`` instance.
  """
  learning_rate = float(params["learning_rate"])

  decay_type = params.get("decay_type", None)
  if decay_type is not None:
    decay_fn = learning_rate_decay_fn(
        decay_type,
        params["decay_rate"],
        params["decay_steps"],
        decay_step_duration=params.get("decay_step_duration", 1),
        staircase=params.get("staircase", True),
        start_decay_steps=params.get("start_decay_steps", 0),
        minimum_learning_rate=params.get("minimum_learning_rate", 0))
    learning_rate = decay_fn(learning_rate, global_step)

  scalar_summary("learning_rate", learning_rate)

  optimizer_class = get_optimizer_class(params["optimizer"])
  optimizer = optimizer_class(learning_rate=learning_rate)
  clip_gradients = params.get("clip_gradients", 0.0)
  if clip_gradients > 0.0:
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clip_gradients)
  if replicated_model:
    optimizer = TowerOptimizer(optimizer)

  return optimizer

def filter_irregular_batches(multiple):
  """Transformation that filters out batches based on their size.

  Args:
    multiple: The divisor of the batch size.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  def _apply_fn(dataset):
    """Transformation function."""

    def _predicate(*x):
      flat = tf.contrib.framework.nest.flatten(x)
      batch_size = tf.shape(flat[0])[0]
      return tf.equal(tf.mod(batch_size, multiple), 0)

    return dataset.filter(_predicate)

  return _apply_fn


@six.add_metaclass(abc.ABCMeta)
class Model(object):
  """Base class for models."""

  def __init__(self, name):
    self.name = name

  def model_fn(self, replicated=False):
    """Model function.

    Args:
      replicated: ``True`` if this model is replicated.

    Returns:
      A model function.

    See Also:
      ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
      arguments and the returned value.
    """
    def _model_fn(features, labels, params, mode, config):
      if mode == tf.estimator.ModeKeys.TRAIN:
        self._register_word_counters(features, labels)

      with tf.variable_scope(self.name, initializer=self._initializer(params)) as model_scope:
        outputs, predictions = self._build(features, labels, params, mode, config)

      if predictions is not None:
        # Register predictions in a collection so that hooks can easily fetch them.
        add_dict_to_collection("predictions", predictions)

      if mode != tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope(model_scope):
          loss = self._compute_loss(features, labels, outputs, params, mode)

        if isinstance(loss, tuple):
          display_loss = loss[1]
          loss = loss[0]
        else:
          display_loss = loss

        scalar_summary("loss", display_loss)

        if mode == tf.estimator.ModeKeys.TRAIN:
          global_step = tf.train.get_or_create_global_step()
          optimizer = get_optimizer(global_step, params, replicated_model=replicated)
          train_op = optimizer.minimize(loss, global_step=global_step)

          return tf.estimator.EstimatorSpec(
              mode, loss=loss, train_op=train_op)
        else:
          eval_metric_ops = self._compute_metrics(features, labels, predictions)

          return tf.estimator.EstimatorSpec(
              mode, loss=loss, eval_metric_ops=eval_metric_ops)
      else:
        export_outputs = {}
        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
            tf.estimator.export.PredictOutput(predictions))

        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions, export_outputs=export_outputs)

    return _model_fn

  def replicated_model_fn(self, num_replicas):
    """Replicate the model accross devices.

    Args:
      num_replicas: The number of replicas.

    Returns:
      A model function.

    Raises:
      ValueError: if :obj:`num_replicas` is greater than the number of visible
        devices.
    """
    devices = [x.name for x in device_lib.list_local_devices() if x.device_type == "GPU"]
    if len(devices) < num_replicas:
      raise ValueError("%d model replicas were requested by only %d devices are visible"
                       % (num_replicas, len(devices)))
    devices = devices[:num_replicas]
    return replicate_model_fn(self.model_fn(replicated=True), devices=devices)

  def _initializer(self, params):
    """Returns the global initializer for this model.

    Args:
      params: A dictionary of hyperparameters.

    Returns:
      The initializer.
    """
    param_init = params.get("param_init")
    if param_init is not None:
      return tf.random_uniform_initializer(minval=-param_init, maxval=param_init)
    return None

  @abc.abstractmethod
  def _build(self, features, labels, params, mode, config):
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
      The loss or a tuple containing the training loss and the loss to display.
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
    """Stores word counter operators for sequences (if any) of :obj:`features`
    and :obj:`labels`.

    See Also:
      :meth:`opennmt.utils.misc.WordCounterHook` that fetches these counters
      to log their value in TensorBoard.
    """
    def _add_counter(word_count, name):
      word_count = tf.cast(word_count, tf.int64)
      counters = tf.get_collection_ref("counters")
      for i, counter in enumerate(counters):
        # If the counter is already registered, update it.
        counter_name = counter.name.split("/")[-1].split(":")[0]
        if counter_name == name:
          counters[i] = tf.assign_add(counter, word_count, name=name)
          return
      total_word_count_init = tf.Variable(
          initial_value=0,
          name=name + "_init",
          trainable=False,
          dtype=tf.int64)
      total_word_count = tf.assign_add(
          total_word_count_init, word_count, name=name)
      counters.append(total_word_count)

    features_length = self._get_features_length(features)
    labels_length = self._get_labels_length(labels)

    with tf.variable_scope("words_per_sec"):
      if features_length is not None:
        _add_counter(tf.reduce_sum(features_length), "features")
      if labels_length is not None:
        _add_counter(tf.reduce_sum(labels_length), "labels")

  def _filter_example(self,
                      features,
                      labels,
                      maximum_features_length=None,
                      maximum_labels_length=None):
    """Defines an example filtering condition.

    Args:
      features: The features ``tf.Tensor``.
      labels: The labels ``tf.Tensor``.
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length of the labels sequence.
        ``None`` to not constrain the length.

    Returns:
      A ``tf.Tensor`` of type ``tf.bool`` with a logical value of ``False``
      if the example does not meet the requirements.
    """
    cond = []

    def _constrain_length(length, maximum_length):
      # Work with lists of lengths which correspond to the general multi source case.
      if not isinstance(length, list):
        length = [length]
      if not isinstance(maximum_length, list):
        maximum_length = [maximum_length]

      # Unset maximum lengths are set to None (i.e. no constraint).
      maximum_length += [None] * (len(length) - len(maximum_length))

      for l, maxlen in zip(length, maximum_length):
        cond.append(tf.greater(l, 0))
        if maxlen is not None:
          cond.append(tf.less_equal(l, maxlen))

    features_length = self._get_features_length(features)
    labels_length = self._get_labels_length(labels)

    if features_length is not None:
      _constrain_length(features_length, maximum_features_length)
    if labels_length is not None:
      _constrain_length(labels_length, maximum_labels_length)

    return tf.reduce_all(cond)

  def _initialize(self, metadata):
    """Runs model specific initialization (e.g. vocabularies loading).

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    pass

  @abc.abstractmethod
  def _get_serving_input_receiver(self):
    """Returns an input receiver for serving this model.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    raise NotImplementedError()

  def _get_features_length(self, features):  # pylint: disable=unused-argument
    """Returns the features length.

    Args:
      features: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor`` or list of ``tf.Tensor``, or ``None`` if
      length is undefined.
    """
    return None

  def _get_labels_length(self, labels):  # pylint: disable=unused-argument
    """Returns the labels length.

    Args:
      labels: A dict of ``tf.Tensor``.

    Returns:
      The length as a ``tf.Tensor``  or ``None`` if length is undefined.
    """
    return None

  @abc.abstractmethod
  def _get_features_builder(self, features_file):
    """Returns the recipe to build features.

    Args:
      features_file: The file of features.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn, padded_shapes_fn)``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_labels_builder(self, labels_file):
    """Returns the recipe to build labels.

    Args:
      labels_file: The file of labels.

    Returns:
      A tuple ``(tf.data.Dataset, process_fn, padded_shapes_fn)``.
    """
    raise NotImplementedError()

  def _input_fn_impl(self,
                     mode,
                     batch_size,
                     prefetch_buffer_size,
                     num_parallel_process_calls,
                     metadata,
                     features_file,
                     labels_file=None,
                     batch_type="examples",
                     batch_multiplier=1,
                     bucket_width=None,
                     sample_buffer_size=None,
                     maximum_features_length=None,
                     maximum_labels_length=None):
    """See ``input_fn``."""
    self._initialize(metadata)

    feat_dataset, feat_process_fn, feat_padded_shapes_fn = self._get_features_builder(features_file)

    if labels_file is None:
      dataset = feat_dataset
      # Parallel inputs must be catched in a single tuple and not considered as multiple arguments.
      process_fn = lambda *arg: feat_process_fn(item_or_tuple(arg))
      padded_shapes_fn = feat_padded_shapes_fn
    else:
      labels_dataset, labels_process_fn, labels_padded_shapes_fn = (
          self._get_labels_builder(labels_file))

      dataset = tf.data.Dataset.zip((feat_dataset, labels_dataset))
      process_fn = lambda features, labels: (
          feat_process_fn(features), labels_process_fn(labels))
      padded_shapes_fn = lambda: (
          feat_padded_shapes_fn(), labels_padded_shapes_fn())

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.shuffle(sample_buffer_size, seed=int(time.time()))

    dataset = dataset.map(
        process_fn,
        num_parallel_calls=num_parallel_process_calls).prefetch(prefetch_buffer_size)
    padded_shapes = padded_shapes_fn()

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.filter(lambda features, labels: self._filter_example(
          features,
          labels,
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length))
      batch_size = batch_size * batch_multiplier

    if mode == tf.estimator.ModeKeys.TRAIN and bucket_width is not None:
      # Form batches with sequences of similar lengths to improve efficiency.
      def _key_func(features, labels):
        features_length = self._get_features_length(features)
        labels_length = self._get_labels_length(labels)

        # For multi inputs, apply bucketing on the target side or none at all.
        if isinstance(features_length, list):
          features_length = None

        bucket_id = tf.constant(0, dtype=tf.int32)
        if features_length is not None:
          bucket_id = tf.maximum(bucket_id, features_length // bucket_width)
        if labels_length is not None:
          bucket_id = tf.maximum(bucket_id, labels_length // bucket_width)
        return tf.to_int64(bucket_id)

      def _reduce_func(unused_key, dataset):
        return dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes)

      def _window_size_func(key):
        if bucket_width > 1:
          key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = batch_size // (key * bucket_width)
        if batch_multiplier > 1:
          # Make the window size a multiple of batch_multiplier.
          size = size + batch_multiplier - size % batch_multiplier
        return tf.to_int64(tf.maximum(size, batch_multiplier))

      if batch_type == "examples":
        batchify_fn = tf.contrib.data.group_by_window(
            _key_func, _reduce_func, window_size=batch_size)
      elif batch_type == "tokens":
        batchify_fn = tf.contrib.data.group_by_window(
            _key_func, _reduce_func, window_size_func=_window_size_func)
      else:
        raise ValueError(
            "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))

      dataset = dataset.apply(batchify_fn)
    else:
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes=padded_shapes)

    if mode == tf.estimator.ModeKeys.TRAIN:
      if batch_multiplier > 1:
        dataset = dataset.apply(filter_irregular_batches(batch_multiplier))
      dataset = dataset.repeat()

    iterator = dataset.make_initializable_iterator()

    # Add the initializer to a standard collection for it to be initialized.
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()

  def input_fn(self,
               mode,
               batch_size,
               prefetch_buffer_size,
               num_parallel_process_calls,
               metadata,
               features_file,
               labels_file=None,
               batch_type="examples",
               batch_multiplier=1,
               bucket_width=None,
               sample_buffer_size=None,
               maximum_features_length=None,
               maximum_labels_length=None):
    """Returns an input function.

    Args:
      mode: A ``tf.estimator.ModeKeys`` mode.
      batch_size: The batch size to use.
      prefetch_buffer_size: The prefetch buffer size.
      num_parallel_process_calls: The number of elements processed in parallel.
      metadata: A dictionary containing additional metadata set
        by the user.
      features_file: The file containing input features.
      labels_file: The file containing output labels.
      batch_type: The training batching stragety to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier to prepare splitting accross
        model replicas.
      bucket_width: The width of the length buckets to select batch candidates
        from. ``None`` to not constrain batch formation.
      sample_buffer_size: The number of elements from which to sample.
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
        prefetch_buffer_size,
        num_parallel_process_calls,
        metadata,
        features_file,
        labels_file=labels_file,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        bucket_width=bucket_width,
        sample_buffer_size=sample_buffer_size,
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
