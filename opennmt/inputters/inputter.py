"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.data import dataset as dataset_util
from opennmt.layers import common
from opennmt.layers.reducer import ConcatReducer, JoinReducer
from opennmt.utils import misc


@six.add_metaclass(abc.ABCMeta)
class Inputter(tf.keras.layers.Layer):
  """Base class for inputters."""

  def __init__(self, **kwargs):
    super(Inputter, self).__init__(**kwargs)
    self.asset_prefix = None

  @property
  def num_outputs(self):
    """The number of parallel outputs produced by this inputter."""
    return 1

  def initialize(self, data_config, asset_prefix=""):
    """Initializes the inputter.

    Args:
      data_config: A dictionary containing the data configuration set
        by the user.
      asset_prefix: The prefix to attach to assets filename.
    """
    _ = data_config
    _ = asset_prefix
    return

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
      A non transformed ``tf.data.Dataset`` instance.
    """
    raise NotImplementedError()

  def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    """Builds a dataset to be used for inference.

    For evaluation and training datasets, see
    :class:`opennmt.inputters.ExampleInputter`.

    Args:
      features_file: The test file.
      batch_size: The batch size to use.
      length_bucket_width: The width of the length buckets to select batch
        candidates from (for efficiency). Set ``None`` to not constrain batch
        formation.
      num_threads: The number of elements processed in parallel.
      prefetch_buffer_size: The number of batches to prefetch asynchronously. If
        ``None``, use an automatically tuned value.

    Returns:
      A ``tf.data.Dataset``.

    See Also:
      :func:`opennmt.data.inference_pipeline`
    """
    map_func = lambda *arg: self.make_features(element=misc.item_or_tuple(arg), training=False)
    dataset = self.make_dataset(features_file, training=False)
    dataset = dataset.apply(dataset_util.inference_pipeline(
        batch_size,
        process_fn=map_func,
        length_bucket_width=length_bucket_width,
        length_fn=self.get_length,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size))
    return dataset

  @abc.abstractmethod
  def input_signature(self):
    """Returns the input signature of this inputter."""
    raise NotImplementedError()

  def get_length(self, features, ignore_special_tokens=False):
    """Returns the length of the input features, if defined.

    Args:
      features: The dictionary of input features.
      ignore_special_tokens: Ignore special tokens that were added by the
        inputter (e.g. <s> and/or </s>).

    Returns:
      The length.
    """
    _ = ignore_special_tokens
    return features.get("length")

  @abc.abstractmethod
  def make_features(self, element=None, features=None, training=None):
    """Creates features from data.

    This is typically called in a data pipeline (such as ``Dataset.map``).
    Common transformation includes tokenization, parsing, vocabulary lookup,
    etc.

    This method accept both a single :obj:`element` from the dataset or a
    partially built dictionary of :obj:`features`.

    Args:
      element: An element from the dataset returned by
        :meth:`opennmt.inputters.Inputter.make_dataset`.
      features: An optional and possibly partial dictionary of features to
        augment.
      training: Run in training mode.

    Returns:
      A dictionary of ``tf.Tensor``.
    """
    raise NotImplementedError()

  def call(self, features, training=None):  # pylint: disable=arguments-differ
    """Creates the model input from the features (e.g. word embeddings).

    Args:
      features: A dictionary of ``tf.Tensor``, the output of
        :meth:`opennmt.inputters.Inputter.make_features`.
      training: Run in training mode.

    Returns:
      The model input.
    """
    _ = training
    return features

  def visualize(self, model_root, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      model_root: The root model object.
      log_dir: The active log directory.
    """
    _ = model_root
    _ = log_dir
    return


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

  def initialize(self, data_config, asset_prefix=""):
    for i, inputter in enumerate(self.inputters):
      inputter.initialize(
          data_config, asset_prefix=_get_asset_prefix(asset_prefix, inputter, i))

  def export_assets(self, asset_dir, asset_prefix=""):
    assets = {}
    for i, inputter in enumerate(self.inputters):
      assets.update(inputter.export_assets(
          asset_dir, asset_prefix=_get_asset_prefix(asset_prefix, inputter, i)))
    return assets

  @abc.abstractmethod
  def make_dataset(self, data_file, training=None):
    raise NotImplementedError()

  def visualize(self, model_root, log_dir):
    for inputter in self.inputters:
      inputter.visualize(model_root, log_dir)


def _get_asset_prefix(prefix, inputter, i):
  return "%s%s_" % (prefix, inputter.asset_prefix or str(i + 1))


class ParallelInputter(MultiInputter):
  """A multi inputter that processes parallel data."""

  def __init__(self,
               inputters,
               reducer=None,
               share_parameters=False,
               combine_features=True):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.Inputter`.
      reducer: A :class:`opennmt.layers.Reducer` to merge all inputs. If
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

  def input_signature(self):
    if self.combine_features:
      signature = {}
      for i, inputter in enumerate(self.inputters):
        for key, value in six.iteritems(inputter.input_signature()):
          signature["{}_{}".format(key, i)] = value
      return signature
    else:
      return tuple(inputter.input_signature() for inputter in self.inputters)

  def get_length(self, features, ignore_special_tokens=False):
    lengths = []
    for i, inputter in enumerate(self.inputters):
      if self.combine_features:
        sub_features = misc.extract_prefixed_keys(features, "inputter_{}_".format(i))
      else:
        sub_features = features[i]
      lengths.append(inputter.get_length(
          sub_features, ignore_special_tokens=ignore_special_tokens))
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
        sub_features = misc.extract_prefixed_keys(features, prefix)
        if not sub_features:
          # Also try to read the format produced by the serving features.
          sub_features = misc.extract_suffixed_keys(features, "_%d" % i)
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
      else:
        features = list(features)
      for i, inputter in enumerate(self.inputters):
        features[i] = inputter.make_features(
            element=element[i] if element is not None else None,
            features=features[i],
            training=training)
      return tuple(features)

  def build(self, input_shape):
    if self.share_parameters:
      # When sharing parameters, build the first leaf inputter and then set
      # all attributes with parameters to the other inputters.
      leaves = self.get_leaf_inputters()
      first, others = leaves[0], leaves[1:]
      first.build(input_shape)
      for name, attr in six.iteritems(first.__dict__):
        if (isinstance(attr, tf.Variable)
            or (isinstance(attr, tf.keras.layers.Layer) and attr.variables)):
          for inputter in others:
            setattr(inputter, name, attr)
            inputter.built = True
    else:
      for inputter in self.inputters:
        inputter.build(input_shape)
    super(ParallelInputter, self).build(input_shape)

  def call(self, features, training=None):
    transformed = []
    for i, inputter in enumerate(self.inputters):
      if self.combine_features:
        sub_features = misc.extract_prefixed_keys(features, "inputter_{}_".format(i))
      else:
        sub_features = features[i]
      transformed.append(inputter(sub_features, training=training))
    if self.reducer is not None:
      transformed = self.reducer(transformed)
    return transformed


class MixedInputter(MultiInputter):
  """An multi inputter that applies several transformation on the same data
  (e.g. combine word-level and character-level embeddings).
  """

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.Inputter`.
      reducer: A :class:`opennmt.layers.Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    super(MixedInputter, self).__init__(inputters, reducer=reducer)
    self.dropout = dropout

  def make_dataset(self, data_file, training=None):
    datasets = [
        inputter.make_dataset(data_file, training=training)
        for inputter in self.inputters]
    for dataset in datasets[1:]:
      if not isinstance(dataset, datasets[0].__class__):
        raise ValueError("All inputters should use the same dataset in a MixedInputter setting")
    return datasets[0]

  def input_signature(self):
    signature = {}
    for inputter in self.inputters:
      signature.update(inputter.input_signature())
    return signature

  def get_length(self, features, ignore_special_tokens=False):
    return self.inputters[0].get_length(features, ignore_special_tokens=ignore_special_tokens)

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    for inputter in self.inputters:
      features = inputter.make_features(
          element=element, features=features, training=training)
    return features

  def build(self, input_shape):
    for inputter in self.inputters:
      inputter.build(input_shape)
    super(MixedInputter, self).build(input_shape)

  def call(self, features, training=None):
    transformed = []
    for inputter in self.inputters:
      transformed.append(inputter(features, training=training))
    outputs = self.reducer(transformed)
    outputs = common.dropout(outputs, self.dropout, training=training)
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
    self.features_inputter.asset_prefix = "source"
    self.labels_inputter = labels_inputter
    self.labels_inputter.asset_prefix = "target"
    super(ExampleInputter, self).__init__(
        [self.features_inputter, self.labels_inputter],
        share_parameters=share_parameters,
        combine_features=False)

  def make_inference_dataset(self,
                             features_file,
                             batch_size,
                             length_bucket_width=None,
                             num_threads=1,
                             prefetch_buffer_size=None):
    return self.features_inputter.make_inference_dataset(
        features_file,
        batch_size,
        length_bucket_width=length_bucket_width,
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
        ``None``, use an automatically tuned value.

    Returns:
      A ``tf.data.Dataset``.

    See Also:
      :func:`opennmt.data.inference_pipeline`
    """
    map_func = lambda *arg: self.make_features(element=arg, training=False)
    dataset = self.make_dataset([features_file, labels_file], training=False)
    dataset = dataset.apply(dataset_util.inference_pipeline(
        batch_size,
        process_fn=map_func,
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size))
    return dataset

  def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            batch_type="examples",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            length_bucket_width=None,
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
        resulting batch size is a multiple of this value.
      shuffle_buffer_size: The number of elements from which to sample.
      length_bucket_width: The width of the length buckets to select batch
        candidates from (for efficiency). Set ``None`` to not constrain batch
        formation.
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
        ``None``, use an automatically tuned value.

    Returns:
      A ``tf.data.Dataset``.

    See Also:
      :func:`opennmt.data.training_pipeline`
    """
    map_func = lambda *arg: self.make_features(element=arg, training=True)
    dataset = self.make_dataset([features_file, labels_file], training=True)
    dataset = dataset.apply(dataset_util.training_pipeline(
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        batch_size_multiple=batch_size_multiple,
        process_fn=map_func,
        length_bucket_width=length_bucket_width,
        features_length_fn=self.features_inputter.get_length,
        labels_length_fn=self.labels_inputter.get_length,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        single_pass=single_pass,
        num_shards=num_shards,
        shard_index=shard_index,
        num_threads=num_threads,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size))
    return dataset
