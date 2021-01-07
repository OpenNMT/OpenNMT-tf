"""Dataset creation and transformations."""

import numpy as np
import tensorflow as tf

from opennmt.utils import misc


def make_datasets(dataset_cls, filenames):
    """Creates instances of :obj:`dataset_cls`.

    Args:
      dataset_cls: A class inheriting from ``tf.data.Dataset``.
      filenames: A list of filenames or a single filename.

    Returns:
      A list of ``tf.data.Dataset`` instances if multiple :obj:`filenames` are
      passed, otherwise a single ``tf.data.Dataset``.

    Raises:
      ValueError: if :obj:`filenames` is empty.
    """
    if not isinstance(filenames, list):
        filenames = [filenames]
    elif not filenames:
        raise ValueError("At least one data file is required")
    datasets = [
        dataset_cls(
            filename, compression_type="GZIP" if misc.is_gzip_file(filename) else None
        )
        for filename in filenames
    ]
    if len(datasets) == 1:
        return datasets[0]
    return datasets


def normalize_weights(datasets, weights=None, sizes=None):
    """Returns normalized dataset weights based on datasets sizes.

    Args:
      datasets: A list of ``tf.data.Dataset`` instances.
      weights: An optional list of dataset weights.
      sizes: The size of each dataset, if known.

    Returns:
      A normalized list of weights that can be used as sampling probabilities.

    Raises:
      ValueError: if the length of :obj:`weights` or :obj:`sizes` does not match
        the length of :obj:`datasets`.
    """
    if not datasets:
        return []
    if len(datasets) == 1:
        return [1.0]

    if weights is None:
        weights = [1 / len(datasets)] * len(datasets)
    elif len(weights) != len(datasets):
        raise ValueError(
            "Got %d datasets but %d weights" % (len(datasets), len(weights))
        )

    if sizes is None:
        sizes = [int(get_dataset_size(dataset)) for dataset in datasets]
    elif len(sizes) != len(datasets):
        raise ValueError("Got %d datasets but %d sizes" % (len(datasets), len(sizes)))

    # Weights should be normalized by the dataset size relative to the total size.
    total_size = sum(sizes)
    weights = [weight * (size / total_size) for weight, size in zip(weights, sizes)]

    # Convert weights to probabilities.
    logits = tf.math.log(tf.constant(weights, dtype=tf.float32))
    probabilities = tf.nn.softmax(logits).numpy().tolist()
    return probabilities


def _get_output_shapes(dataset):
    """Returns the outputs shapes of the dataset.

    Args:
      dataset: A ``tf.data.Dataset``.

    Returns:
      A nested structure of ``tf.TensorShape``
    """
    return tf.nest.map_structure(lambda spec: spec.shape, dataset.element_spec)


def get_dataset_size(dataset, batch_size=5000):
    """Get the dataset size.

    Example:

      >>> dataset = tf.data.Dataset.range(5)
      >>> opennmt.data.get_dataset_size(dataset).numpy()
      5

    Args:
      dataset: A dataset.
      batch_size: The batch size to use to improve the scan performance, or
        ``None`` to scan the dataset as-is.

    Returns:
      The dataset size or ``None`` if the dataset is infinite.
    """
    if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
        return None
    if batch_size is not None:
        dataset = dataset.batch(batch_size)

    def _reduce_func(count, element):
        element = tf.nest.flatten(element)[0]
        batch_size = tf.shape(element)[0]
        return count + tf.cast(batch_size, count.dtype)

    return dataset.reduce(tf.constant(0, dtype=tf.int64), _reduce_func)


def filter_irregular_batches(multiple):
    """Transformation that filters out batches based on their size.

    Example:

      >>> dataset = tf.data.Dataset.range(10).batch(3)
      >>> dataset = dataset.apply(opennmt.data.filter_irregular_batches(3))
      >>> len(list(iter(dataset)))
      3

    Args:
      multiple: The divisor of the batch size.

    Returns:
      A ``tf.data.Dataset`` transformation.
    """
    if multiple == 1:
        return lambda dataset: dataset

    def _predicate(*x):
        flat = tf.nest.flatten(x)
        batch_size = tf.shape(flat[0])[0]
        return tf.equal(batch_size % multiple, 0)

    return lambda dataset: dataset.filter(_predicate)


def filter_examples_by_length(
    maximum_features_length=None,
    maximum_labels_length=None,
    features_length_fn=None,
    labels_length_fn=None,
):
    """Transformation that filters out examples with zero length or length that is
    greater than the configured maximum.

    Example:

      >>> dataset = dataset.apply(opennmt.data.filter_examples_by_length(...))

    Args:
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length or list of maximum lengths of
        the labels sequence(s). ``None`` to not constrain the length.
      features_length_fn: A function mapping features to a sequence length.
      labels_length_fn: A function mapping labels to a sequence length.

    Returns:
      A ``tf.data.Dataset`` transformation.
    """
    if features_length_fn is None and labels_length_fn is None:
        return lambda dataset: dataset

    def _length_constraints(lengths, maximum_lengths):
        # Work with lists of lengths which correspond to the general multi source case.
        if not isinstance(lengths, list):
            lengths = [lengths]
        if not isinstance(maximum_lengths, list):
            maximum_lengths = [maximum_lengths]
        # Unset maximum lengths are set to None (i.e. no constraint).
        maximum_lengths += [None] * (len(lengths) - len(maximum_lengths))
        constraints = []
        for length, maximum_length in zip(lengths, maximum_lengths):
            constraints.append(tf.greater(length, 0))
            if maximum_length is not None:
                constraints.append(tf.less_equal(length, maximum_length))
        return constraints

    def _predicate(features, labels):
        cond = []
        features_length = (
            features_length_fn(features) if features_length_fn is not None else None
        )
        labels_length = (
            labels_length_fn(labels) if labels_length_fn is not None else None
        )
        if features_length is not None:
            cond.extend(_length_constraints(features_length, maximum_features_length))
        if labels_length is not None:
            cond.extend(_length_constraints(labels_length, maximum_labels_length))
        return tf.reduce_all(cond)

    return lambda dataset: dataset.filter(_predicate)


def make_cardinality_multiple_of(divisor):
    """Transformation that ensures that the dataset cardinality is a multiple of
    :obj:`divisor`.

    Example:

      >>> dataset = tf.data.Dataset.range(7)
      >>> dataset = dataset.apply(opennmt.data.make_cardinality_multiple_of(10))
      >>> len(list(iter(dataset)))
      10

    Args:
      divisor: The value that should divide the dataset size.

    Returns:
      A ``tf.data.Dataset`` transformation.

    Tip:
      This transformation is useful when training multiple replicas on a finite
      dataset. It ensures that each replica receives a non empty batch in the last
      training iteration.
    """
    if divisor == 1:
        return lambda dataset: dataset

    def _continue_iter(num_consumed, element):
        # Continue iterating if the current element is from the original dataset or
        # if the number of consumed batches is not a multiple of divisor.
        is_original = element[0]
        return tf.math.logical_or(
            is_original, tf.math.not_equal(num_consumed % divisor, 0)
        )

    def _retrieve_element(num_consumed, element):
        _ = num_consumed
        return element[1]

    def _transform(dataset):
        # Nothing to do for infinite datasets.
        if dataset.cardinality() == tf.data.INFINITE_CARDINALITY:
            return dataset

        # Concatenate extra batches with a flag.
        extra_batches = dataset.repeat()
        dataset = dataset.map(lambda *x: (tf.constant(True), x))
        extra_batches = extra_batches.map(lambda *x: (tf.constant(False), x))
        dataset = dataset.concatenate(extra_batches)

        # Take all original batches and the number of extra batches required.
        dataset = dataset.enumerate()
        dataset = dataset.apply(tf.data.experimental.take_while(_continue_iter))
        return dataset.map(_retrieve_element)  # Retrieve the element only.

    return _transform


def random_shard(shard_size, dataset_size):
    """Transformation that shards the dataset in a random order.

    Example:

      >>> dataset = tf.data.Dataset.range(6)
      >>> dataset = dataset.apply(opennmt.data.random_shard(2, 6))
      >>> list(dataset.as_numpy_iterator())
      [0, 1, 4, 5, 2, 3]

    Args:
      shard_size: The number of examples in each shard.
      dataset_size: The total number of examples in the dataset.

    Returns:
      A ``tf.data.Dataset`` transformation.
    """
    num_shards = -(-dataset_size // shard_size)  # Ceil division.
    offsets = np.linspace(
        0, dataset_size, num=num_shards, endpoint=False, dtype=np.int64
    )

    def _random_shard(dataset):
        sharded_dataset = tf.data.Dataset.from_tensor_slices(offsets)
        sharded_dataset = sharded_dataset.shuffle(num_shards)
        sharded_dataset = sharded_dataset.flat_map(
            lambda offset: dataset.skip(offset).take(shard_size)
        )
        return sharded_dataset

    return _random_shard


def shuffle_dataset(buffer_size, shuffle_shards=True, dataset_size=None):
    """Transformation that shuffles the dataset based on its size.

    Example:

      >>> dataset = tf.data.Dataset.range(6)
      >>> dataset = dataset.apply(opennmt.data.shuffle_dataset(3))
      >>> list(dataset.as_numpy_iterator())
      [2, 3, 1, 0, 4, 5]

    Args:
      buffer_size: The number of elements from which to sample.
      shuffle_shards: When :obj:`buffer_size` is smaller than the dataset size,
        the dataset is first sharded in a random order to add another level of
        shuffling.
      dataset_size: If the dataset size is already known, it can be passed here to
        avoid a slower generic computation of the dataset size later.

    Returns:
      A ``tf.data.Dataset`` transformation.
    """

    def _shuffle(dataset):
        sample_size = buffer_size
        if sample_size < 0 or shuffle_shards:
            total_size = dataset_size
            if total_size is None:
                total_size = get_dataset_size(dataset)
            tf.get_logger().info("Training on %d examples", total_size)
            if sample_size < 0:
                sample_size = total_size
            elif sample_size < total_size:
                dataset = dataset.apply(random_shard(sample_size, total_size))
        dataset = dataset.shuffle(sample_size)
        return dataset

    return _shuffle


def batch_dataset(batch_size, padded_shapes=None):
    """Transformation that batches a dataset.

    Example:

      >>> dataset = dataset.apply(opennmt.data.batch_dataset(...))

    Args:
      batch_size: The batch size.
      padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
        are automatically inferred from the dataset output shapes.

    Returns:
      A ``tf.data.Dataset`` transformation.

    See Also:
      :func:`opennmt.data.batch_sequence_dataset`
    """
    return lambda dataset: dataset.padded_batch(
        batch_size, padded_shapes=padded_shapes or _get_output_shapes(dataset)
    )


def batch_sequence_dataset(
    batch_size,
    batch_type="examples",
    batch_multiplier=1,
    batch_size_multiple=1,
    length_bucket_width=None,
    length_fn=None,
    padded_shapes=None,
):
    """Transformation that batches a dataset of sequences.

    This implements an example-based and a token-based batching strategy
    with optional bucketing of sequences.

    Bucketing makes the batches contain sequences of similar lengths to optimize
    the training efficiency. For example, if :obj:`length_bucket_width` is 5,
    sequences will be organized by the following length buckets:

    1 - 5 | 6 - 10 | 11 - 15 | ...

    Then when building the next batch, sequences will be selected from the same
    length bucket.

    If the dataset has parallel elements (e.g. a parallel source and target
    dataset), the element is assigned to the bucket corresponding to the maximum
    length of all parallel elements.

    Example:

      >>> dataset = dataset.apply(opennmt.data.batch_sequence_dataset(...))

    Args:
      batch_size: The batch size.
      batch_type: The training batching strategy to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier.
      batch_size_multiple: When :obj:`batch_type` is "tokens", ensure that the
        resulting batch size is a multiple of this value.
      length_bucket_width: The width of the length buckets to select batch
        candidates from. ``None`` to not constrain batch formation.
      length_fn: A function or list of functions (in case of a parallel dataset)
        that take features as argument and return the associated sequence length.
      padded_shapes: The padded shapes for this dataset. If ``None``, the shapes
        are automatically inferred from the dataset output shapes.

    Returns:
      A ``tf.data.Dataset`` transformation.

    Raises:
      ValueError: if :obj:`batch_type` is not one of "examples" or "tokens".
      ValueError: if :obj:`batch_type` is "tokens" but :obj:`length_bucket_width`
        is not set.
      ValueError: if the number of length functions in :obj:`length_fn` does not
        match the number of parallel elements.

    See Also:
      :func:`opennmt.data.batch_dataset`
    """
    batch_size = batch_size * batch_multiplier

    def _get_bucket_id(features, length_fn):
        default_id = tf.constant(0, dtype=tf.int32)
        if length_fn is None:
            return default_id
        lengths = length_fn(features)
        if lengths is None:
            return default_id
        if not isinstance(lengths, list):
            lengths = [lengths]  # Fallback to the general case of parallel inputs.
        lengths = [length // length_bucket_width for length in lengths]
        return tf.reduce_max(lengths)

    def _key_func(*args):
        length_fns = length_fn
        if length_fns is None:
            length_fns = [None for _ in args]
        elif not isinstance(length_fns, (list, tuple)):
            length_fns = [length_fns]
        if len(length_fns) != len(args):
            raise ValueError(
                "%d length functions were passed but this dataset contains "
                "%d parallel elements" % (len(length_fns), len(args))
            )
        # Take the highest bucket id.
        bucket_id = tf.reduce_max(
            [
                _get_bucket_id(features, length_fn)
                for features, length_fn in zip(args, length_fns)
            ]
        )
        return tf.cast(bucket_id, tf.int64)

    def _reduce_func(unused_key, dataset):
        return dataset.apply(batch_dataset(batch_size, padded_shapes=padded_shapes))

    def _window_size_func(key):
        if length_bucket_width > 1:
            key += 1  # For length_bucket_width == 1, key 0 is unassigned.
        size = batch_size // (key * length_bucket_width)
        required_multiple = batch_multiplier * batch_size_multiple
        if required_multiple > 1:
            size = size + required_multiple - size % required_multiple
        return tf.cast(tf.maximum(size, required_multiple), tf.int64)

    if length_bucket_width is None:
        if batch_type == "tokens":
            raise ValueError(
                "Batch type 'tokens' requires length bucketing (the parameter "
                "length_bucket_width should be non null)"
            )
        return batch_dataset(batch_size, padded_shapes=padded_shapes)

    if batch_type == "examples":
        return tf.data.experimental.group_by_window(
            _key_func, _reduce_func, window_size=batch_size
        )
    elif batch_type == "tokens":
        return tf.data.experimental.group_by_window(
            _key_func, _reduce_func, window_size_func=_window_size_func
        )
    else:
        raise ValueError(
            "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(
                batch_type
            )
        )


def training_pipeline(
    batch_size,
    batch_type="examples",
    batch_multiplier=1,
    batch_size_multiple=1,
    process_fn=None,
    transform_fns=None,
    length_bucket_width=None,
    features_length_fn=None,
    labels_length_fn=None,
    maximum_features_length=None,
    maximum_labels_length=None,
    single_pass=False,
    num_shards=1,
    shard_index=0,
    num_threads=None,
    dataset_size=None,
    shuffle_buffer_size=None,
    prefetch_buffer_size=None,
    cardinality_multiple=1,
):
    """Transformation that applies most of the dataset operations commonly used
    for training on sequence data:

    * sharding
    * shuffling
    * processing
    * filtering
    * bucketization
    * batching
    * prefetching

    Example:

      >>> dataset = dataset.apply(opennmt.data.training_pipeline(...))

    Args:
      batch_size: The batch size to use.
      batch_type: The training batching strategy to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier.
      batch_size_multiple: When :obj:`batch_type` is "tokens", ensure that the
        resulting batch size is a multiple of this value.
      process_fn: The processing function to apply on each element.
      transform_fns: List of dataset transformation functions (applied after
        :obj:`process_fn` if defined).
      length_bucket_width: The width of the length buckets to select batch
        candidates from. ``None`` to not constrain batch formation.
      features_length_fn: A function mapping features to a sequence length.
      labels_length_fn: A function mapping labels to a sequence length.
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length of the labels sequence.
        ``None`` to not constrain the length.
      single_pass: If ``True``, makes a single pass over the training data.
      num_shards: The number of data shards (usually the number of workers in a
        distributed setting).
      shard_index: The shard index this data pipeline should read from.
      num_threads: The number of elements processed in parallel.
      dataset_size: If the dataset size is already known, it can be passed here to
        avoid a slower generic computation of the dataset size later.
      shuffle_buffer_size: The number of elements from which to sample.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value.
      cardinality_multiple: Ensure that the dataset cardinality is a multiple of
        this value when :obj:`single_pass` is ``True``.

    Returns:
      A ``tf.data.Dataset`` transformation.

    See Also:
      - :func:`opennmt.data.batch_sequence_dataset`
      - :func:`opennmt.data.make_cardinality_multiple_of`
      - :func:`opennmt.data.filter_examples_by_length`
      - :func:`opennmt.data.filter_irregular_batches`
      - :func:`opennmt.data.shuffle_dataset`
    """
    if dataset_size is not None and num_shards > 1:
        # Update dataset_size based on the shard size.
        if isinstance(dataset_size, list):
            dataset_size = [size // num_shards for size in dataset_size]
        else:
            dataset_size //= num_shards

    def _make_weighted_dataset(datasets, weights):
        if single_pass:
            raise ValueError(
                "single_pass parameter is not compatible with weighted datasets"
            )
        if not datasets:
            raise ValueError("At least one dataset is required")
        if weights is not None and len(weights) != len(datasets):
            raise ValueError(
                "%d dataset weights were provided, but %d were expected to match the "
                "number of data files" % (len(weights), len(datasets))
            )
        if num_shards > 1:
            datasets = [dataset.shard(num_shards, shard_index) for dataset in datasets]
        weights = normalize_weights(datasets, weights=weights, sizes=dataset_size)
        datasets = [dataset.repeat() for dataset in datasets]
        dataset = tf.data.experimental.sample_from_datasets(datasets, weights=weights)
        if shuffle_buffer_size is not None and shuffle_buffer_size != 0:
            if shuffle_buffer_size < 0:
                raise ValueError(
                    "shuffle_buffer_size < 0 is not compatible with weighted datasets"
                )
            dataset = dataset.shuffle(shuffle_buffer_size)
        return dataset

    def _make_single_dataset(dataset):
        if num_shards > 1:
            dataset = dataset.shard(num_shards, shard_index)
        if shuffle_buffer_size is not None and shuffle_buffer_size != 0:
            dataset = dataset.apply(
                shuffle_dataset(shuffle_buffer_size, dataset_size=dataset_size)
            )
        return dataset

    def _pipeline(dataset):
        if isinstance(dataset, tuple):
            dataset, weights = dataset
        else:
            weights = None
        is_weighted_dataset = isinstance(dataset, list)
        if is_weighted_dataset:
            dataset = _make_weighted_dataset(dataset, weights)
        else:
            dataset = _make_single_dataset(dataset)
        if process_fn is not None:
            dataset = dataset.map(process_fn, num_parallel_calls=num_threads or 4)
        if transform_fns is not None:
            for transform_fn in transform_fns:
                dataset = dataset.apply(transform_fn)
        dataset = dataset.apply(
            filter_examples_by_length(
                maximum_features_length=maximum_features_length,
                maximum_labels_length=maximum_labels_length,
                features_length_fn=features_length_fn,
                labels_length_fn=labels_length_fn,
            )
        )
        dataset = dataset.apply(
            batch_sequence_dataset(
                batch_size,
                batch_type=batch_type,
                batch_multiplier=batch_multiplier,
                batch_size_multiple=batch_size_multiple,
                length_bucket_width=length_bucket_width,
                length_fn=[features_length_fn, labels_length_fn],
            )
        )
        dataset = dataset.apply(filter_irregular_batches(batch_multiplier))
        if not single_pass:
            if not is_weighted_dataset:  # Weighted dataset is repeated before sampling.
                dataset = dataset.repeat()
        else:
            dataset = dataset.apply(make_cardinality_multiple_of(cardinality_multiple))
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    return _pipeline


def inference_pipeline(
    batch_size,
    batch_type="examples",
    process_fn=None,
    transform_fns=None,
    length_bucket_width=None,
    length_fn=None,
    num_threads=None,
    prefetch_buffer_size=None,
):
    """Transformation that applies dataset operations for inference.

    Example:

      >>> dataset = dataset.apply(opennmt.data.inference_pipeline(...))

    Args:
      batch_size: The batch size to use.
      batch_type: The batching strategy to use: can be "examples" or "tokens".
      process_fn: The processing function to apply on each element.
      transform_fns: List of dataset transformation functions (applied after
        :obj:`process_fn` if defined).
      length_bucket_width: The width of the length buckets to select batch
        candidates from. If set, this means the inference pipeline will be
        reordered based on the examples length, the application is then
        responsible to restore the predictions in order. An "index" key will be
        inserted in the examples dictionary.
      length_fn: A function mapping features to a sequence length.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value.

    Returns:
      A ``tf.data.Dataset`` transformation.

    Raises:
      ValueError: if :obj:`length_bucket_width` is set but not :obj:`length_fn`.
      ValueError: if :obj:`length_bucket_width` is set but the dataset does not
        output a dictionary structure.
    """

    def _inject_index(index, x):
        if isinstance(x, tuple):
            features = x[0]
        else:
            features = x
        features["index"] = index
        return x

    def _pipeline(dataset):
        if process_fn is not None:
            dataset = dataset.map(process_fn, num_parallel_calls=num_threads)
        if transform_fns is not None:
            for transform_fn in transform_fns:
                dataset = dataset.apply(transform_fn)
        if length_bucket_width is not None and length_bucket_width > 0:
            if length_fn is None:
                raise ValueError("length_fn is required when reordering by length")
            output_shapes = _get_output_shapes(dataset)
            if isinstance(output_shapes, tuple):
                num_length_fn = (
                    len(length_fn) if isinstance(length_fn, (list, tuple)) else 1
                )
                if len(output_shapes) != num_length_fn:
                    raise ValueError(
                        "The dataset outputs %d parallel features, but got %d "
                        "length functions" % (len(output_shapes), num_length_fn)
                    )
                output_shapes = output_shapes[0]
            if not isinstance(output_shapes, dict):
                raise ValueError(
                    "Reordering by length expects dataset elements to be Python dicts"
                )
            dataset = dataset.enumerate()
            dataset = dataset.map(_inject_index)
            dataset = dataset.apply(
                batch_sequence_dataset(
                    batch_size,
                    batch_type=batch_type,
                    length_bucket_width=length_bucket_width,
                    length_fn=length_fn,
                )
            )
        else:
            dataset = dataset.apply(batch_dataset(batch_size))
        dataset = dataset.prefetch(prefetch_buffer_size)
        return dataset

    return _pipeline
