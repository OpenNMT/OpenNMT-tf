"""Base class for models."""

from __future__ import print_function

import time
import abc
import six

import tensorflow as tf

from opennmt.utils import decay
from opennmt.utils.misc import add_dict_to_collection


def learning_rate_decay_fn(decay_type,
                           decay_rate,
                           decay_steps,
                           staircase=True,
                           start_decay_steps=0,
                           minimum_learning_rate=0):
  """Returns the learning rate decay functions.

  Args:
    decay_type: The type of decay. A function from `tf.train` or `opennmt.utils.decay`
      as a `String`.
    decay_rate: The decay rate to apply.
    decay_steps: The decay steps as described in the decay type function.
    staircase: If `True`, learning rate is decayed in a staircase fashion.
    start_decay_steps: Start decay after this many steps.
    minimum_learning_rate: Do not decay past this learning rate value.

  Returns:
    A function with signature `lambda learning_rate, global_steps: decayed_learning_rate`.

  Raises:
    ValueError: if `decay_type` can not be resolved.
  """
  def _decay_fn(learning_rate, global_step):
    decay_op_name = None

    if decay_op_name is None:
      decay_op_name = getattr(tf.train, decay_type, None)
    if decay_op_name is None:
      decay_op_name = getattr(decay, decay_type, None)
    if decay_op_name is None:
      raise ValueError("Unknown decay function: {}".format(decay_type))

    decayed_learning_rate = decay_op_name(
        learning_rate,
        tf.maximum(global_step - start_decay_steps, 0),
        decay_steps,
        decay_rate,
        staircase=staircase)
    decayed_learning_rate = tf.maximum(decayed_learning_rate, minimum_learning_rate)

    return decayed_learning_rate

  return _decay_fn


@six.add_metaclass(abc.ABCMeta)
class Model(object):
  """Base class for models."""

  def __init__(self, name):
    self.name = name

  def __call__(self, features, labels, params, mode, config):
    """Creates the model. See `tf.estimator.Estimator`'s `model_fn` argument
    for more details about arguments and the returned value.
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
      self._register_word_counters(features, labels)

    with tf.variable_scope(self.name):
      outputs, predictions = self._build(features, labels, params, mode, config)

      if predictions is not None:
        # Register predictions in a collection so that hooks can easily fetch them.
        add_dict_to_collection(predictions, "predictions")

      if mode != tf.estimator.ModeKeys.PREDICT:
        loss = self._compute_loss(features, labels, outputs)
        train_op = self._build_train_op(loss, params)

        if mode == tf.estimator.ModeKeys.EVAL:
          eval_metric_ops = self._compute_metrics(features, labels, predictions)
        else:
          eval_metric_ops = None

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops)
      else:
        export_outputs = {}
        export_outputs[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.estimator.export.PredictOutput(predictions)

        return tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            export_outputs=export_outputs)

  @abc.abstractmethod
  def _build(self, features, labels, params, mode, config):
    """Creates the graph.

    See `tf.estimator.Estimator`'s `model_fn` for arguments description.

    Returns:
      outputs: The model outputs (usually unscaled probabilities).
        Optional if `mode` is `tf.estimator.ModeKeys.PREDICT`.
      predictions: The model predictions.
        Optional if `mode` is `tf.estimator.ModeKeys.TRAIN`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _compute_loss(self, features, labels, outputs):
    """Computes the loss.

    Args:
      features: The dict of features `tf.Tensor`s.
      labels: The dict of labels `tf.Tensor`s.
      output: The model outputs (usually unscaled probabilities).

    Returns:
      The loss.
    """
    raise NotImplementedError()

  def _compute_metrics(self, features, labels, predictions):  # pylint: disable=unused-argument
    """Computes additional metrics on the predictions.

    Args:
      features: The dict of features `tf.Tensor`s.
      labels: The dict of labels `tf.Tensor`s.
      predictions: The model predictions.

    Returns:
      A dict of metric results (tuple `(metric_tensor, update_op)`) keyed by name.
    """
    return None

  def _build_train_op(self, loss, params):
    """Builds the training op given parameters."""
    global_step = tf.train.get_or_create_global_step()
    decay_type = params.get("decay_type")

    if decay_type is not None:
      decay_fn = learning_rate_decay_fn(
          decay_type,
          params["decay_rate"],
          params["decay_steps"],
          staircase=params.get("staircase", True),
          start_decay_steps=params.get("start_decay_steps", 0),
          minimum_learning_rate=params.get("minimum_learning_rate", 0))
    else:
      decay_fn = None

    train_op = tf.contrib.layers.optimize_loss(
        loss,
        global_step,
        params["learning_rate"],
        params["optimizer"],
        clip_gradients=params.get("clip_gradients"),
        learning_rate_decay_fn=decay_fn,
        summaries=[
            "learning_rate",
            "loss",
            "global_gradient_norm",
        ])

    return train_op

  def _register_word_counters(self, features, labels):
    """Stores word counter operators for sequences (if any) of `features`
    and `labels`.

    See also `onmt.utils.misc.WordCounterHook` that fetches these counters
    to log their value in TensorBoard.
    """
    def _add_counter(word_count, name):
      word_count = tf.cast(word_count, tf.int64)
      total_word_count_init = tf.Variable(
          initial_value=0,
          name=name + "_init",
          trainable=False,
          dtype=tf.int64)
      total_word_count = tf.assign_add(
          total_word_count_init,
          word_count,
          name=name)
      tf.add_to_collection("counters", total_word_count)

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
      features: A dict of `tf.Tensor`s.
      labels: A `tf.Tensor` or dict of `tf.Tensor`s.
      maximum_features_length: The maximum length of the features
        sequence (if it applies).
      maximum_labels_length: The maximum length of the labels sequence
        (if it applies).

    Returns:
      A `tf.Tensor` of type `tf.bool` with a logical value of `False`
      if the example does not meet the requirements.
    """
    features_length = self._get_features_length(features)
    labels_length = self._get_labels_length(labels)

    cond = []

    if features_length is not None:
      cond.append(tf.greater(features_length, 0))
      if maximum_features_length is not None:
        cond.append(tf.less_equal(features_length, maximum_features_length))

    if labels_length is not None:
      cond.append(tf.greater(labels_length, 0))
      if maximum_labels_length is not None:
        cond.append(tf.less_equal(labels_length, maximum_labels_length))

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
      A `tf.estimator.export.ServingInputReceiver`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_features_length(self, features):
    """Returns the features length.

    Args:
      features: A dict of `tf.Tensor`s

    Returns:
      The length as a `tf.Tensor`, or `None` if length is undefined.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_labels_length(self, labels):
    """Returns the labels length.

    Args:
      labels: A dict of `tf.Tensor`s

    Returns:
      The length as a `tf.Tensor`, or `None` if length is undefined.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_features_builder(self, features_file):
    """Returns the recipe to build features.

    Args:
      features_file: The file of features.

    Returns:
      A tuple (`tf.data.Dataset`, `process_fn`, `padded_shapes_fn`)
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _get_labels_builder(self, labels_file):
    """Returns the recipe to build labels.

    Args:
      labels_file: The file of labels.

    Returns:
      A tuple (`tf.data.Dataset`, `process_fn`, `padded_shapes_fn`)
    """
    raise NotImplementedError()

  def _input_fn_impl(self,
                     mode,
                     batch_size,
                     buffer_size,
                     num_parallel_process_calls,
                     metadata,
                     features_file,
                     labels_file=None,
                     num_buckets=None,
                     maximum_features_length=None,
                     maximum_labels_length=None):
    """See `input_fn`."""
    self._initialize(metadata)

    feat_dataset, feat_process_fn, feat_padded_shapes_fn = self._get_features_builder(features_file)

    if labels_file is None:
      dataset = feat_dataset
      process_fn = feat_process_fn
      padded_shapes_fn = feat_padded_shapes_fn
    else:
      labels_dataset, labels_process_fn, labels_padded_shapes_fn = (
          self._get_labels_builder(labels_file))

      dataset = tf.data.Dataset.zip((feat_dataset, labels_dataset))
      process_fn = lambda features, labels: (
          feat_process_fn(features), labels_process_fn(labels))
      padded_shapes_fn = lambda: (
          feat_padded_shapes_fn(), labels_padded_shapes_fn())

    dataset = dataset.map(
        process_fn,
        num_parallel_calls=num_parallel_process_calls).prefetch(buffer_size)
    padded_shapes = padded_shapes_fn()

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.filter(lambda features, labels: self._filter_example(
          features,
          labels,
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length))
      dataset = dataset.shuffle(buffer_size, seed=int(time.time()))
      dataset = dataset.repeat()

    if mode == tf.estimator.ModeKeys.PREDICT or num_buckets is None or num_buckets <= 1:
      dataset = dataset.padded_batch(
          batch_size,
          padded_shapes=padded_shapes)
    else:
      # For training and evaluation, use bucketing.

      def _key_func(features, unused_labels):
        if maximum_features_length:
          bucket_width = (maximum_features_length + num_buckets - 1) // num_buckets
        else:
          bucket_width = 10

        bucket_id = self._get_features_length(features) // bucket_width
        bucket_id = tf.minimum(bucket_id, num_buckets)
        return tf.to_int64(bucket_id)

      def _reduce_func(unused_key, dataset):
        return dataset.padded_batch(
            batch_size,
            padded_shapes=padded_shapes)

      dataset = dataset.apply(tf.contrib.data.group_by_window(
          key_func=_key_func,
          reduce_func=_reduce_func,
          window_size=batch_size))

    iterator = dataset.make_initializable_iterator()

    # Add the initializer to a standard collection for it to be initialized.
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()

  def input_fn(self,
               mode,
               batch_size,
               buffer_size,
               num_parallel_process_calls,
               metadata,
               features_file,
               labels_file=None,
               num_buckets=None,
               maximum_features_length=None,
               maximum_labels_length=None):
    """Returns an input function.

    See also `tf.estimator.Estimator`.

    Args:
      mode: A `tf.estimator.ModeKeys` mode.
      batch_size: The batch size to use.
      buffer_size: The prefetch buffer size (used e.g. for shuffling).
      num_parallel_process_calls: The number of elements processed in parallel.
      metadata: A dictionary containing additional metadata set
        by the user.
      features_file: The file containing input features.
      labels_file: The file containing output labels.
      num_buckets: The number of buckets to store examples of similar sizes.
      maximum_features_length: The maximum length of feature sequences
        during training (if it applies).
      maximum_labels_length: The maximum length of label sequences
        during training (if it applies).

    Returns:
      A callable that returns the next element.

    Raises:
      ValueError: if `labels_file` is not set when in training or evaluation mode.
    """
    if mode != tf.estimator.ModeKeys.PREDICT and labels_file is None:
      raise ValueError("Labels file is required for training and evaluation")

    return lambda: self._input_fn_impl(
        mode,
        batch_size,
        buffer_size,
        num_parallel_process_calls,
        metadata,
        features_file,
        labels_file=labels_file,
        num_buckets=num_buckets,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length)

  def _serving_input_fn_impl(self, metadata):
    """See `serving_input_fn`."""
    self._initialize(metadata)
    return self._get_serving_input_receiver()

  def serving_input_fn(self, metadata):
    """Returns the serving input function.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.

    Returns:
      A callable that returns a `tf.estimator.export.ServingInputReceiver`.
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
