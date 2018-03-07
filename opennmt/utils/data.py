"""Functions for reading data."""

import tensorflow as tf
import numpy as np


def filter_irregular_batches(multiple):
  """Transformation that filters out batches based on their size.

  Args:
    multiple: The divisor of the batch size.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  if multiple == 1:
    return lambda dataset: dataset

  def _predicate(*x):
    flat = tf.contrib.framework.nest.flatten(x)
    batch_size = tf.shape(flat[0])[0]
    return tf.equal(tf.mod(batch_size, multiple), 0)

  return lambda dataset: dataset.filter(_predicate)

def filter_examples_by_length(maximum_features_length=None,
                              maximum_labels_length=None,
                              features_length_fn=None,
                              labels_length_fn=None):
  """Transformation that constrains examples length.

  Args:
    maximum_features_length: The maximum length or list of maximum lengths of
      the features sequence(s). ``None`` to not constrain the length.
    maximum_labels_length: The maximum length of the labels sequence.
      ``None`` to not constrain the length.
    features_length_fn: A callable mapping features to a sequence length.
    labels_length_fn: A callable mapping labels to a sequence length.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  if features_length_fn is None and labels_length_fn is None:
    return lambda dataset: dataset

  def _length_constraints(length, maximum_length):
    # Work with lists of lengths which correspond to the general multi source case.
    if not isinstance(length, list):
      length = [length]
    if not isinstance(maximum_length, list):
      maximum_length = [maximum_length]
    # Unset maximum lengths are set to None (i.e. no constraint).
    maximum_length += [None] * (len(length) - len(maximum_length))
    constraints = []
    for l, maxlen in zip(length, maximum_length):
      constraints.append(tf.greater(l, 0))
      if maxlen is not None:
        constraints.append(tf.less_equal(l, maxlen))
    return constraints

  def _predicate(features, labels):
    cond = []
    features_length = features_length_fn(features) if features_length_fn is not None else None
    labels_length = labels_length_fn(labels) if labels_length_fn is not None else None
    if features_length is not None:
      cond.extend(_length_constraints(features_length, maximum_features_length))
    if labels_length is not None:
      cond.extend(_length_constraints(labels_length, maximum_labels_length))
    return tf.reduce_all(cond)

  return lambda dataset: dataset.filter(_predicate)

def random_shard(shard_size, dataset_size):
  """Transformation that shards the dataset in a random order.

  Args:
    shard_size: The number of examples in each shard.
    dataset_size: The total number of examples in the dataset.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  num_shards = -(-dataset_size // shard_size)  # Ceil division.
  offsets = np.linspace(0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64)

  def _random_shard(dataset):
    sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
    sharded_dataset = sharded_dataset.shuffle(num_shards)
    sharded_dataset = sharded_dataset.flat_map(
        lambda offset: dataset.skip(offset).take(shard_size))
    return sharded_dataset

  return _random_shard

def batch_train_dataset(batch_size,
                        batch_type="examples",
                        batch_multiplier=1,
                        bucket_width=None,
                        padded_shapes=None,
                        features_length_fn=None,
                        labels_length_fn=None):
  """Transformation that batches the dataset for training.

  This implements an example-based and a token-based batching strategy
  with optional bucketing of sequences.

  Bucketing makes the batches contain sequences of similar lengths to optimize
  the training efficiency. For example, if :obj:`bucket_width` is 5, sequences
  will be organized by lengths:

  1 - 5 | 6 - 10 | 11 - 15 | ...

  where the assigned length is the maximum of the source and target lengths.
  Then each batch will only consider sequences from the same bucket.

  Args:
    batch_size: The batch size.
    batch_type: The training batching stragety to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier to prepare splitting accross
      replicated graph parts.
    bucket_width: The sequence length bucket width.
    padded_shapes: The padded shapes.
    features_length_fn: A callable mapping features to a sequence length.
    labels_length_fn: A callable mapping labels to a sequence length.

  Returns:
    A ``tf.data.Dataset`` transformation.

  Raises:
    ValueError: if :obj:`batch_type` is not one of "examples" or "tokens".
  """
  batch_size = batch_size * batch_multiplier

  if bucket_width is None:
    return lambda dataset: dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes)

  def _key_func(features, labels):
    features_length = features_length_fn(features) if features_length_fn is not None else None
    labels_length = labels_length_fn(labels) if labels_length_fn is not None else None
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
    return tf.contrib.data.group_by_window(
        _key_func, _reduce_func, window_size=batch_size)
  elif batch_type == "tokens":
    return tf.contrib.data.group_by_window(
        _key_func, _reduce_func, window_size_func=_window_size_func)
  else:
    raise ValueError(
        "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))
