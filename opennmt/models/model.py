"""Base class for models."""

import tensorflow as tf

import abc
import six


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

    train_op = tf.contrib.layers.optimize_loss(
      loss,
      global_step,
      params["learning_rate"],
      params["optimizer"],
      clip_gradients=params.get("clip_gradients"))

    return train_op

  def _filter_example(self, features, labels):
    """Defines an example filtering condition."""
    return True

  @abc.abstractmethod
  def _get_size(self, features, labels):
    """Defines a size to an example for data bucketing."""
    raise NotImplementedError()

  def _get_maximum_size(self):
    """Defines the maximum size of an example for data bucketing."""
    return None

  @abc.abstractmethod
  def _build_dataset(self, mode, batch_size, features_file, labels_file=None):
    """Builds a dataset from features and labels files.

    Args:
      mode: A `tf.estimator.ModeKeys` mode.
      batch_size: The batch size to use.
      features_file: The file of features.
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
                     labels_file=None):
    """See `input_fn`."""
    dataset, padded_shapes = self._build_dataset(
      mode,
      batch_size,
      features_file,
      labels_file=labels_file)

    if mode == tf.estimator.ModeKeys.TRAIN:
      dataset = dataset.filter(lambda features, labels: self._filter_example(features, labels))
      dataset = dataset.shuffle(buffer_size=buffer_size)
      dataset = dataset.repeat()
    elif mode == tf.estimator.ModeKeys.EVAL:
      dataset = dataset.repeat()

    if mode == tf.estimator.ModeKeys.PREDICT:
      dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes)
    else:
      # For training and evaluation, use bucketing.

      def key_func(features, labels):
        maximum_size = self._get_maximum_size()

        if maximum_size:
          bucket_width = (maximum_size + num_buckets - 1) // num_buckets
        else:
          bucket_width = 10

        bucket_id = self._get_size(features, labels) // bucket_width
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
               labels_file=None):
    """Returns an input function.

    See also `tf.estimator.Estimator`.

    Args:
      mode: A `tf.estimator.ModeKeys` mode.
      batch_size: The batch size to use.
      buffer_size: The prefetch buffer size (used e.g. for shuffling).
      num_buckets: The number of buckets to store examples of similar sizes.
      features_file: The file containing input features.
      labels_file: The file containing output labels.

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
      labels_file=labels_file)

  def format_prediction(self, prediction, params=None):
    """Formats the model prediction.

    Args:
      prediction: The evaluated prediction returned by `__call__`.
      params: (optional) Dictionary of formatting parameters.

    Returns:
      The final prediction.
    """
    return prediction
