"""Functions for reading data."""

import tensorflow as tf
import numpy as np


def get_padded_shapes(dataset):
  """Returns the padded shapes for ``tf.data.Dataset.padded_batch``.

  Args:
    dataset: The dataset that will be batched with padding.

  Returns:
    The same structure as ``dataset.output_shapes`` containing the padded
    shapes.
  """
  return tf.contrib.framework.nest.map_structure(
      lambda shape: shape.as_list(), dataset.output_shapes)

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

def prefetch_element(buffer_size=None):
  """Transformation that prefetches elements from the dataset.

  This is a small wrapper around tf.data.Dataset.prefetch to customize the
  case :obj:`buffer_size` is ``None`` for some TensorFlow versions.

  Args:
    buffer_size: The number of batches to prefetch asynchronously. If ``None``,
      use an automatically tuned value on TensorFlow 1.8+ and 1 on older
      versions.

  Returns:
    A ``tf.data.Dataset`` transformation.
  """
  if not hasattr(tf.contrib.data, "AUTOTUNE") and buffer_size is None:
    buffer_size = 1
  return lambda dataset: dataset.prefetch(buffer_size)

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

def batch_parallel_dataset(batch_size,
                           batch_type="examples",
                           batch_multiplier=1,
                           bucket_width=None,
                           padded_shapes=None,
                           features_length_fn=None,
                           labels_length_fn=None):
  """Transformation that batches a parallel dataset.

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
    batch_type: The training batching strategy to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier to prepare splitting accross
      replicated graph parts.
    bucket_width: The sequence length bucket width.
    padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
      are automatically inferred from the dataset output shapes.
    features_length_fn: A callable mapping features to a sequence length.
    labels_length_fn: A callable mapping labels to a sequence length.

  Returns:
    A ``tf.data.Dataset`` transformation.

  Raises:
    ValueError: if :obj:`batch_type` is not one of "examples" or "tokens".
  """
  batch_size = batch_size * batch_multiplier

  def _batch_func(dataset):
    return dataset.padded_batch(
        batch_size,
        padded_shapes=padded_shapes or get_padded_shapes(dataset))

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
    return _batch_func(dataset)

  def _window_size_func(key):
    if bucket_width > 1:
      key += 1  # For bucket_width == 1, key 0 is unassigned.
    size = batch_size // (key * bucket_width)
    if batch_multiplier > 1:
      # Make the window size a multiple of batch_multiplier.
      size = size + batch_multiplier - size % batch_multiplier
    return tf.to_int64(tf.maximum(size, batch_multiplier))

  if bucket_width is None:
    return _batch_func

  if batch_type == "examples":
    return tf.contrib.data.group_by_window(
        _key_func, _reduce_func, window_size=batch_size)
  elif batch_type == "tokens":
    return tf.contrib.data.group_by_window(
        _key_func, _reduce_func, window_size_func=_window_size_func)
  else:
    raise ValueError(
        "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))


def training_pipeline(dataset,
                      batch_size,
                      batch_type="examples",
                      batch_multiplier=1,
                      bucket_width=None,
                      single_pass=False,
                      process_fn=None,
                      num_threads=None,
                      shuffle_buffer_size=None,
                      prefetch_buffer_size=None,
                      dataset_size=None,
                      maximum_features_length=None,
                      maximum_labels_length=None,
                      features_length_fn=None,
                      labels_length_fn=None):
  """Defines a complete training data pipeline.

  Args:
    dataset: The base dataset.
    batch_size: The batch size to use.
    batch_type: The training batching stragety to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier to prepare splitting accross
       replicated graph parts.
    bucket_width: The width of the length buckets to select batch candidates
      from. ``None`` to not constrain batch formation.
    single_pass: If ``True``, makes a single pass over the training data.
    process_fn: The processing function to apply on each element.
    num_threads: The number of elements processed in parallel.
    shuffle_buffer_size: The number of elements from which to sample.
    prefetch_buffer_size: The number of batches to prefetch asynchronously. If
      ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
      older versions.
    dataset_size: The total size of the dataset, if known. It is recommended to
      set it when :obj:`shuffle_buffer_size` is smaller than the dataset size.
    maximum_features_length: The maximum length or list of maximum lengths of
      the features sequence(s). ``None`` to not constrain the length.
    maximum_labels_length: The maximum length of the labels sequence.
      ``None`` to not constrain the length.
    features_length_fn: A callable mapping features to a sequence length.
    labels_length_fn: A callable mapping labels to a sequence length.

  Returns:
    A ``tf.data.Dataset``.
  """
  if shuffle_buffer_size is not None and shuffle_buffer_size != 0:
    if dataset_size is not None:
      if shuffle_buffer_size < 0:
        shuffle_buffer_size = dataset_size
      elif shuffle_buffer_size < dataset_size:
        # When the sample buffer size is smaller than the dataset size, shard
        # the dataset in a random order. This ensures that all parts of the
        # dataset can be seen when the evaluation frequency is high.
        dataset = dataset.apply(random_shard(shuffle_buffer_size, dataset_size))
    dataset = dataset.shuffle(shuffle_buffer_size)
  if process_fn is not None:
    dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 4)
  dataset = dataset.apply(filter_examples_by_length(
      maximum_features_length=maximum_features_length,
      maximum_labels_length=maximum_labels_length,
      features_length_fn=features_length_fn,
      labels_length_fn=labels_length_fn))
  dataset = dataset.apply(batch_parallel_dataset(
      batch_size,
      batch_type=batch_type,
      batch_multiplier=batch_multiplier,
      bucket_width=bucket_width,
      features_length_fn=features_length_fn,
      labels_length_fn=labels_length_fn))
  dataset = dataset.apply(filter_irregular_batches(batch_multiplier))
  if not single_pass:
    dataset = dataset.repeat()
  dataset = dataset.apply(prefetch_element(buffer_size=prefetch_buffer_size))
  return dataset

def inference_pipeline(dataset,
                       batch_size,
                       process_fn=None,
                       num_threads=None,
                       prefetch_buffer_size=None):
  """Defines a complete inference data pipeline.

  Args:
    dataset: The base dataset.
    batch_size: The batch size to use.
    process_fn: The processing function to apply on each element.
    num_threads: The number of elements processed in parallel.
    prefetch_buffer_size: The number of batches to prefetch asynchronously. If
      ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
      older versions.

  Returns:
    A ``tf.data.Dataset``.
  """
  if process_fn is not None:
    dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 1)
  dataset = dataset.apply(batch_parallel_dataset(batch_size))
  dataset = dataset.apply(prefetch_element(buffer_size=prefetch_buffer_size))
  return dataset
