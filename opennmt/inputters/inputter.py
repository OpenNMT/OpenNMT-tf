"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import ConcatReducer, JoinReducer
from opennmt.utils import compat
from opennmt.utils.data import inference_pipeline, training_pipeline
from opennmt.utils.misc import extract_prefixed_keys, extract_suffixed_keys, item_or_tuple


@six.add_metaclass(abc.ABCMeta)
class Inputter(tf.keras.layers.Layer):
  """Base class for inputters."""

  def __init__(self, dtype=tf.float32):
    super(Inputter, self).__init__(dtype=dtype)
    self.volatile = set()
    self.process_hooks = []
    self.is_target = False

  @property
  def num_outputs(self):
    """How many parallel outputs does this inputter produce."""
    return 1

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    """Initializes the inputter.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
      asset_dir: The directory where assets can be written. If ``None``, no
        assets are returned.
      asset_prefix: The prefix to attach to assets filename.

    Returns:
      A dictionary containing additional assets used by the inputter.
    """
    _ = metadata
    if asset_dir is not None:
      return self.export_assets(asset_dir, asset_prefix=asset_prefix)
    return {}

  def export_assets(self, asset_dir, asset_prefix=""):
    """Exports assets used by this tokenizer.

    Args:
      asset_dir: The directory where assets can be written.
      asset_prefix: The prefix to attach to assets filename.

    Returns:
      A dictionary containing additional assets used by the inputter.
    """
    _ = asset_dir
    _ = asset_prefix
    return {}

  @abc.abstractmethod
  def make_dataset(self, data_file, training=None):
    """Creates the base dataset required by this inputter.

    Args:
      data_file: The data file.
      training: Run in training mode.

    Returns:
      A ``tf.data.Dataset``.
    """
    raise NotImplementedError()

  def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    """Builds a dataset to be used for inference.

    For evaluation and training datasets, see
    :class:`opennmt.inputters.inputter.ExampleInputter`.

    Args:
      features_file: The test file.
      batch_size: The batch size to use.
      bucket_width: The width of the length buckets to select batch candidates
        from (for efficiency). Set ``None`` to not constrain batch formation.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
        older versions.

    Returns:
      A ``tf.data.Dataset``.
    """
    map_func = lambda *arg: self.make_features(item_or_tuple(arg), training=False)
    dataset = self.make_dataset(features_file, training=False)
    dataset = inference_pipeline(
        dataset,
        batch_size,
        process_fn=map_func,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size,
        bucket_width=bucket_width,
        length_fn=self.get_length)
    return dataset

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    """Returns the size of the dataset.

    Args:
      data_file: The data file.

    Returns:
      The total size.
    """
    raise NotImplementedError()

  def get_serving_input_receiver(self):
    """Returns a serving input receiver for this inputter.

    Returns:
      A ``tf.estimator.export.ServingInputReceiver``.
    """
    if self.is_target:
      raise ValueError("Target inputters do not define a serving input")
    receiver_tensors = self.get_receiver_tensors()
    if receiver_tensors is None:
      raise NotImplementedError("This inputter does not define receiver tensors.")
    features = self.make_features(features=receiver_tensors.copy())
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def get_receiver_tensors(self):
    """Returns the input placeholders for serving."""
    return None

  def get_length(self, features):
    """Returns the length of the input features, if defined."""
    return features.get("length")

  @abc.abstractmethod
  def make_features(self, element=None, features=None, training=None):
    """Creates features from data.

    Args:
      element: An element from the dataset.
      features: An optional dictionary of features to augment.
      training: Run in training mode.

    Returns:
      A dictionary of ``tf.Tensor``.
    """
    raise NotImplementedError()

  def call(self, features, training=None):  # pylint: disable=arguments-differ
    """Forwards call to ``make_inputs().``"""
    return self.make_inputs(features, training=training)

  def make_inputs(self, features, training=None):
    """Creates the model input from the features.

    Args:
      features: A dictionary of ``tf.Tensor``.
      training: Run in training mode.

    Returns:
      The model input.
    """
    _ = training
    return features

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    _ = log_dir
    return


  # TODO: remove the following methods at some point.

  def set_data_field(self, data, key, value, volatile=False):
    """Sets a data field.

    Args:
      data: The data dictionary.
      key: The value key.
      value: The value to assign.
      volatile: If ``True``, the key/value pair will be removed once the
        processing done.

    Returns:
      The updated data dictionary.
    """
    data[key] = value
    if volatile:
      self.volatile.add(key)
    return data

  def remove_data_field(self, data, key):
    """Removes a data field.

    Args:
      data: The data dictionary.
      key: The value key.

    Returns:
      The updated data dictionary.
    """
    del data[key]
    return data

  def add_process_hooks(self, hooks):
    """Adds processing hooks.

    Processing hooks are additional and model specific data processing
    functions applied after calling this inputter
    :meth:`opennmt.inputters.inputter.Inputter.process` function.

    Args:
      hooks: A list of callables with the signature
        ``(inputter, data) -> data``.
    """
    self.process_hooks.extend(hooks)

  def process(self, data, training=None):
    """Prepares raw data.

    Args:
      data: The raw data.
      training: Run in training mode.

    Returns:
      A dictionary of ``tf.Tensor``.

    See Also:
      :meth:`opennmt.inputters.inputter.Inputter.transform_data`
    """
    data = self.make_features(data, training=training)
    for hook in self.process_hooks:
      data = hook(self, data)
    for key in self.volatile:
      data = self.remove_data_field(data, key)
    self.volatile.clear()
    return data

  def transform_data(self, data, mode=tf.estimator.ModeKeys.TRAIN, log_dir=None):
    """Transforms the processed data to an input.

    This is usually a simple forward of a :obj:`data` field to
    :meth:`opennmt.inputters.inputter.Inputter.transform`.

    See also `process`.

    Args:
      data: A dictionary of data fields.
      mode: A ``tf.estimator.ModeKeys`` mode.
      log_dir: The log directory. If set, visualization will be setup.

    Returns:
      The transformed input.
    """
    inputs = self.make_inputs(data, training=mode == tf.estimator.ModeKeys.TRAIN)
    if log_dir:
      self.visualize(log_dir)
    return inputs


@six.add_metaclass(abc.ABCMeta)
class MultiInputter(Inputter):
  """An inputter that gathers multiple inputters, possibly nested."""

  def __init__(self, inputters, reducer=None):
    if not isinstance(inputters, list) or not inputters:
      raise ValueError("inputters must be a non empty list")
    dtype = inputters[0].dtype
    for inputter in inputters:
      if inputter.dtype != dtype:
        raise TypeError("All inputters must have the same dtype")
    super(MultiInputter, self).__init__(dtype=dtype)
    self.inputters = inputters
    self.reducer = reducer

  @property
  def num_outputs(self):
    if self.reducer is None or isinstance(self.reducer, JoinReducer):
      return len(self.inputters)
    return 1

  def get_leaf_inputters(self):
    """Returns a list of all leaf Inputter instances."""
    inputters = []
    for inputter in self.inputters:
      if isinstance(inputter, MultiInputter):
        inputters.extend(inputter.get_leaf_inputters())
      else:
        inputters.append(inputter)
    return inputters

  def __getattribute__(self, name):
    if name == "built":
      return all(inputter.built for inputter in self.inputters)
    else:
      return super(MultiInputter, self).__getattribute__(name)

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    for i, inputter in enumerate(self.inputters):
      inputter.initialize(metadata, asset_prefix="%s%d_" % (asset_prefix, i + 1))
    return super(MultiInputter, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)

  def export_assets(self, asset_dir, asset_prefix=""):
    assets = {}
    for i, inputter in enumerate(self.inputters):
      assets.update(inputter.export_assets(
          asset_dir, asset_prefix="%s%d_" % (asset_prefix, i + 1)))
    return assets

  @abc.abstractmethod
  def make_dataset(self, data_file, training=None):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    raise NotImplementedError()

  def visualize(self, log_dir):
    for inputter in self.inputters:
      inputter.visualize(log_dir)


class ParallelInputter(MultiInputter):
  """An multi inputter that process parallel data."""

  def __init__(self,
               inputters,
               reducer=None,
               share_parameters=False,
               combine_features=True):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs. If
        set, parallel inputs are assumed to have the same length.
      share_parameters: Share the inputters parameters.
      combine_features: Combine each inputter features in a single dict or
        return them separately. This is typically ``True`` for multi source
        inputs but ``False`` for features/labels parallel data.

    Raises:
      ValueError: if :obj:`share_parameters` is set but the child inputters are
        not of the same type.
    """
    super(ParallelInputter, self).__init__(inputters, reducer=reducer)
    self.combine_features = combine_features
    self.share_parameters = share_parameters
    if self.share_parameters:
      leaves = self.get_leaf_inputters()
      for inputter in leaves[1:]:
        if type(inputter) is not type(leaves[0]):
          raise ValueError("Each inputter must be of the same type for parameter sharing")

  def make_dataset(self, data_file, training=None):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    datasets = [
        inputter.make_dataset(data, training=training)
        for inputter, data in zip(self.inputters, data_file)]
    return tf.data.Dataset.zip(tuple(datasets))

  def get_dataset_size(self, data_file):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    dataset_sizes = [
        inputter.get_dataset_size(data)
        for inputter, data in zip(self.inputters, data_file)]
    dataset_size = dataset_sizes[0]
    for size in dataset_sizes:
      if size != dataset_size:
        raise RuntimeError("The parallel data files do not have the same size")
    return dataset_size

  def get_receiver_tensors(self):
    receiver_tensors = {}
    for i, inputter in enumerate(self.inputters):
      tensors = inputter.get_receiver_tensors()
      for key, value in six.iteritems(tensors):
        receiver_tensors["{}_{}".format(key, i)] = value
    return receiver_tensors

  def get_length(self, features):
    lengths = []
    for i, inputter in enumerate(self.inputters):
      if self.combine_features:
        sub_features = extract_prefixed_keys(features, "inputter_{}_".format(i))
      else:
        sub_features = features[i]
      lengths.append(inputter.get_length(sub_features))
    if self.reducer is None:
      return lengths
    else:
      return lengths[0]

  def make_features(self, element=None, features=None, training=None):
    if self.combine_features:
      if features is None:
        features = {}
      for i, inputter in enumerate(self.inputters):
        prefix = "inputter_%d_" % i
        sub_features = extract_prefixed_keys(features, prefix)
        if not sub_features:
          # Also try to read the format produced by get_receiver_tensors.
          sub_features = extract_suffixed_keys(features, "_%d" % i)
        sub_features = inputter.make_features(
            element=element[i] if element is not None else None,
            features=sub_features,
            training=training)
        for key, value in six.iteritems(sub_features):
          features["%s%s" % (prefix, key)] = value
      return features
    else:
      if features is None:
        features = [{} for _ in self.inputters]
      for i, inputter in enumerate(self.inputters):
        features[i] = inputter.make_features(
            element=element[i] if element is not None else None,
            features=features[i],
            training=training)
      return tuple(features)

  def _get_names(self):
    for i, _ in enumerate(self.inputters):
      yield "inputter_%d" % i

  def _get_shared_name(self):
    return ""

  def _get_scopes(self):
    for _, name in zip(self.inputters, self._get_names()):
      if self.share_parameters:
        name = self._get_shared_name()
      yield name

  def build(self, input_shape=None):
    if self.share_parameters:
      # When sharing parameters, build the first leaf inputter and then set
      # all attributes with parameters to the other inputters.
      leaves = self.get_leaf_inputters()
      first, others = leaves[0], leaves[1:]
      with compat.tf_compat(v1="variable_scope")(self._get_shared_name()):
        first.build(input_shape)
      for name, attr in six.iteritems(first.__dict__):
        if (isinstance(attr, tf.Variable)
            or (isinstance(attr, tf.keras.layers.Layer) and attr.variables)):
          for inputter in others:
            setattr(inputter, name, attr)
            inputter.built = True
    else:
      for inputter, scope in zip(self.inputters, self._get_names()):
        with compat.tf_compat(v1="variable_scope")(scope):
          inputter.build(input_shape)
    super(ParallelInputter, self).build(input_shape)

  def make_inputs(self, features, training=None):
    if not self.built:
      self.build()
    transformed = []
    for i, (inputter, scope) in enumerate(zip(self.inputters, self._get_scopes())):
      with compat.tf_compat(v1="variable_scope")(scope):
        if self.combine_features:
          sub_features = extract_prefixed_keys(features, "inputter_{}_".format(i))
        else:
          sub_features = features[i]
        transformed.append(inputter.make_inputs(sub_features, training=training))
    if self.reducer is not None:
      transformed = self.reducer(transformed)
    return transformed


class MixedInputter(MultiInputter):
  """An multi inputter that applies several transformation on the same data."""

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    super(MixedInputter, self).__init__(inputters, reducer=reducer)
    self.dropout = dropout

  def make_dataset(self, data_file, training=None):
    datasets = [
        inputter.make_dataset(data_file, training=training)
        for inputter in self.inputters]
    return datasets[0]

  def get_dataset_size(self, data_file):
    return self.inputters[0].get_dataset_size(data_file)

  def get_receiver_tensors(self):
    receiver_tensors = {}
    for inputter in self.inputters:
      receiver_tensors.update(inputter.get_receiver_tensors())
    return receiver_tensors

  def get_length(self, features):
    return self.inputters[0].get_length(features)

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    for inputter in self.inputters:
      features = inputter.make_features(
          element=element, features=features, training=training)
    return features

  def make_inputs(self, features, training=None):
    transformed = []
    for i, inputter in enumerate(self.inputters):
      with compat.tf_compat(v1="variable_scope")("inputter_{}".format(i)):
        transformed.append(inputter.make_inputs(features, training=training))
    outputs = self.reducer(transformed)
    outputs = tf.layers.dropout(outputs, rate=self.dropout, training=training)
    return outputs


class ExampleInputter(ParallelInputter):
  """An inputter that returns training examples (parallel features and labels)."""

  def __init__(self, features_inputter, labels_inputter, share_parameters=False):
    """Initializes this inputter.

    Args:
      features_inputter: An inputter producing the features (source).
      labels_inputter: An inputter producing the labels (target).
      share_parameters: Share the inputters parameters.
    """
    self.features_inputter = features_inputter
    self.labels_inputter = labels_inputter
    self.labels_inputter.is_target = True
    super(ExampleInputter, self).__init__(
        [self.features_inputter, self.labels_inputter],
        share_parameters=share_parameters,
        combine_features=False)

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    self.features_inputter.initialize(metadata, asset_prefix="source_")
    self.labels_inputter.initialize(metadata, asset_prefix="target_")
    if asset_dir is not None:
      return self.export_assets(asset_dir, asset_prefix=asset_prefix)
    return {}

  def export_assets(self, asset_dir, asset_prefix=""):
    assets = {}
    assets.update(self.features_inputter.export_assets(
        asset_dir, asset_prefix="source_"))
    assets.update(self.labels_inputter.export_assets(
        asset_dir, asset_prefix="target_"))
    return assets

  def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    return self.features_inputter.make_inference_dataset(
        features_file,
        batch_size,
        bucket_width=bucket_width,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size)

  def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              num_threads=1,
                              prefetch_buffer_size=None):
    """Builds a dataset to be used for evaluation.

    Args:
      features_file: The evaluation source file.
      labels_file: The evaluation target file.
      batch_size: The batch size to use.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
        older versions.

    Returns:
      A ``tf.data.Dataset``.
    """
    map_func = lambda *arg: self.make_features(arg, training=False)
    dataset = self.make_dataset([features_file, labels_file], training=False)
    dataset = inference_pipeline(
        dataset,
        batch_size,
        process_fn=map_func,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size)
    return dataset

  def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            batch_type="examples",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
    """Builds a dataset to be used for training. It supports the full training
    pipeline, including:

    * sharding
    * shuffling
    * filtering
    * bucketing
    * prefetching

    Args:
      features_file: The evaluation source file.
      labels_file: The evaluation target file.
      batch_size: The batch size to use.
      batch_type: The training batching stragety to use: can be "examples" or
        "tokens".
      batch_multiplier: The batch size multiplier to prepare splitting accross
         replicated graph parts.
      batch_size_multiple: When :obj:`batch_type` is "tokens", ensure that the
        result batch size is a multiple of this value.
      shuffle_buffer_size: The number of elements from which to sample.
      bucket_width: The width of the length buckets to select batch candidates
        from (for efficiency). Set ``None`` to not constrain batch formation.
      maximum_features_length: The maximum length or list of maximum lengths of
        the features sequence(s). ``None`` to not constrain the length.
      maximum_labels_length: The maximum length of the labels sequence.
        ``None`` to not constrain the length.
      single_pass: If ``True``, makes a single pass over the training data.
      num_shards: The number of data shards (usually the number of workers in a
        distributed setting).
      shard_index: The shard index this data pipeline should read from.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
        older versions.

    Returns:
      A ``tf.data.Dataset``.
    """
    dataset_size = self.features_inputter.get_dataset_size(features_file)
    map_func = lambda *arg: self.make_features(arg, training=True)
    dataset = self.make_dataset([features_file, labels_file], training=True)
    dataset = training_pipeline(
        dataset,
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        bucket_width=bucket_width,
        single_pass=single_pass,
        process_fn=map_func,
        num_threads=num_threads,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size,
        dataset_size=dataset_size,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        features_length_fn=self.features_inputter.get_length,
        labels_length_fn=self.labels_inputter.get_length,
        batch_size_multiple=batch_size_multiple,
        num_shards=num_shards,
        shard_index=shard_index)
    return dataset
