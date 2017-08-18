"""Base class for models."""

import abc
import six
import time

import tensorflow as tf


def learning_rate_decay_fn(decay_type,
                           decay_rate,
                           decay_steps,
                           staircase=True,
                           start_decay_steps=0,
                           minimum_learning_rate=0):
  """Returns the learning rate decay functions.

  Args:
    decay_type: The type of decay. A function from `tf.train` as a `String`.
    decay_rate: The decay rate to apply.
    decay_steps: The decay steps as described in the decay type function.
    staircase: If `True`, learning rate is decayed in a staircase fashion.
    start_decay_steps: Start decay after this many steps.
    minimum_learning_rate: Do not decay past this learning rate value.

  Returns:
    A function with signature `lambda learning_rate, global_steps: decayed_learning_rate`.
  """
  def decay_fn(learning_rate, global_step):
    decay_class = getattr(tf.train, decay_type)

    decayed_learning_rate = decay_class(
      learning_rate,
      tf.maximum(global_step - start_decay_steps, 0),
      decay_steps,
      decay_rate,
      staircase=staircase)
    decayed_learning_rate = tf.maximum(decayed_learning_rate, minimum_learning_rate)

    return decayed_learning_rate

  return decay_fn


@six.add_metaclass(abc.ABCMeta)
class Model(object):

  def __init__(self, name):
    self.name = name

  def __call__(self, features, labels, params, mode):
    """Creates the model. See `tf.estimator.Estimator`'s `model_fn` argument
    for more details about arguments and the returned value.
    """
    with tf.variable_scope(self.name):
      return self._build(features, labels, params, mode)

  @abc.abstractmethod
  def _build(self, features, labels, params, mode):
    """Creates the model. Subclasses should override this function."""
    raise NotImplementedError()

  def _build_train_op(self, loss, params):
    """Builds the training op given parameters."""
    global_step = tf.train.get_or_create_global_step()

    if "decay_type" in params:
      decay_fn = learning_rate_decay_fn(
        params.get("decay_type"),
        params.get("decay_rate"),
        params.get("decay_steps"),
        staircase=params.get("staircase") or True,
        start_decay_steps=params.get("start_decay_steps") or 0,
        minimum_learning_rate=params.get("minimum_learning_rate") or 0)
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

  def _filter_example(self,
                      features,
                      labels,
                      maximum_features_length=None,
                      maximum_labels_length=None):
    """Defines an example filtering condition."""
    features_length = self._features_length(features)
    labels_length = self._labels_length(labels)

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

  def _features_length(self, features):
    """Attributes a length to a feature (if defined)."""
    return None

  def _labels_length(self, labels):
    """Attributes a length to a label (if defined)."""
    return None

  @abc.abstractmethod
  def _build_features(self, features_file):
    """Builds a dataset from features file.

    Args:
      features_file: The file of features.

    Returns:
      (`tf.contrib.data.Dataset`, `padded_shapes`)
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def _build_labels(self, labels_file):
    """Builds a dataset from labels file.

    Args:
      labels_file: The file of labels.

    Returns:
      (`tf.contrib.data.Dataset`, `padded_shapes`)
    """
    raise NotImplementedError()

  def _input_fn_impl(self,
                     mode,
                     batch_size,
                     buffer_size,
                     num_buckets,
                     features_file,
                     labels_file=None,
                     maximum_features_length=None,
                     maximum_labels_length=None):
    """See `input_fn`."""
    features_dataset, features_padded_shapes = self._build_features(features_file)

    if labels_file is None:
      dataset = features_dataset
      padded_shapes = features_padded_shapes
    else:
      labels_dataset, labels_padded_shapes = self._build_labels(labels_file)
      dataset = tf.contrib.data.Dataset.zip((features_dataset, labels_dataset))
      padded_shapes = (features_padded_shapes, labels_padded_shapes)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.filter(lambda features, labels: self._filter_example(
        features,
        labels,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length))
      dataset = dataset.shuffle(buffer_size, seed=int(time.time()))
      dataset = dataset.repeat()

    if mode == tf.estimator.ModeKeys.PREDICT or num_buckets <= 1:
      dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes)
    else:
      # For training and evaluation, use bucketing.

      def key_func(features, labels):
        if maximum_features_length:
          bucket_width = (maximum_features_length + num_buckets - 1) // num_buckets
        else:
          bucket_width = 10

        bucket_id = self._features_length(features) // bucket_width
        bucket_id = tf.minimum(bucket_id, num_buckets)
        return tf.to_int64(bucket_id)

      def reduce_func(key, dataset):
        return dataset.padded_batch(
          batch_size,
          padded_shapes=padded_shapes)

      dataset = dataset.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=batch_size)

    iterator = dataset.make_initializable_iterator()

    # Add the initializer to a standard collection for it to be initialized.
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)

    return iterator.get_next()

  def input_fn(self,
               mode,
               batch_size,
               buffer_size,
               num_buckets,
               features_file,
               labels_file=None,
               maximum_features_length=None,
               maximum_labels_length=None):
    """Returns an input function.

    See also `tf.estimator.Estimator`.

    Args:
      mode: A `tf.estimator.ModeKeys` mode.
      batch_size: The batch size to use.
      buffer_size: The prefetch buffer size (used e.g. for shuffling).
      num_buckets: The number of buckets to store examples of similar sizes.
      features_file: The file containing input features.
      labels_file: The file containing output labels.
      maximum_features_length: The maximum length of feature sequences
        during training (if it applies).
      maximum_labels_length: The maximum length of label sequences
        during training (if it applies).

    Returns:
      A callable that returns the next element.
    """
    if mode != tf.estimator.ModeKeys.PREDICT and labels_file is None:
      raise ValueError("Labels file is required for training and evaluation")

    return lambda: self._input_fn_impl(
      mode,
      batch_size,
      buffer_size,
      num_buckets,
      features_file,
      labels_file=labels_file,
      maximum_features_length=maximum_features_length,
      maximum_labels_length=maximum_labels_length)

  def format_prediction(self, prediction, params=None):
    """Formats the model prediction.

    Args:
      prediction: The evaluated prediction returned by `__call__`.
      params: (optional) Dictionary of formatting parameters.

    Returns:
      The final prediction.
    """
    return prediction
