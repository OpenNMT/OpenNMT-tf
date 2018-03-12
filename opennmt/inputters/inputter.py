"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import ConcatReducer
from opennmt.utils.misc import extract_prefixed_keys


@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  """Base class for inputters."""

  def __init__(self, dtype=tf.float32):
    self.volatile = set()
    self.process_hooks = []
    self.dtype = dtype

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

  def get_length(self, unused_data):
    """Returns the length of the input data, if defined."""
    return None

  @abc.abstractmethod
  def make_dataset(self, data_file):
    """Creates the dataset required by this inputter.

    Args:
      data_file: The data file.

    Returns:
      A ``tf.data.Dataset``.
    """
    raise NotImplementedError()

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
    receiver_tensors, features = self._get_serving_input()
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _get_serving_input(self):
    """Returns the input receiver for serving.

    Returns:
      A tuple ``(receiver_tensors, features)`` as described in
      ``tf.estimator.export.ServingInputReceiver``.
    """
    raise NotImplementedError()

  def initialize(self, metadata):
    """Initializes the inputter within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    ``TABLE_INITIALIZERS`` collection.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    pass

  def process(self, data):
    """Prepares raw data.

    Args:
      data: The raw data.

    Returns:
      A dictionary of ``tf.Tensor``.

    See Also:
      :meth:`opennmt.inputters.inputter.Inputter.transform_data`
    """
    data = self._process(data)
    for hook in self.process_hooks:
      data = hook(self, data)
    for key in self.volatile:
      data = self.remove_data_field(data, key)
    self.volatile.clear()
    return data

  def _process(self, data):
    """Prepares raw data (implementation).

    Subclasses should extend this function to prepare the raw value read
    from the dataset to something they can transform (e.g. processing a
    line of text to a sequence of ids).

    This base implementation makes sure the data is a dictionary so subclasses
    can populate it.

    Args:
      data: The raw data or a dictionary containing the ``raw`` key.

    Returns:
      A dictionary of ``tf.Tensor``.

    Raises:
      ValueError: if :obj:`data` is a dictionary but does not contain the
        ``raw`` key.
    """
    if not isinstance(data, dict):
      data = self.set_data_field({}, "raw", data, volatile=True)
    elif "raw" not in data:
      raise ValueError("data must contain the raw dataset value")
    return data

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    pass

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
    inputs = self._transform_data(data, mode)
    if log_dir:
      self.visualize(log_dir)
    return inputs

  @abc.abstractmethod
  def _transform_data(self, data, mode):
    """Implementation of ``transform_data``."""
    raise NotImplementedError()

  @abc.abstractmethod
  def transform(self, inputs, mode):
    """Transforms inputs.

    Args:
      inputs: A (possible nested structure of) ``tf.Tensor`` which depends on
        the inputter.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      The transformed input.
    """
    raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class MultiInputter(Inputter):
  """An inputter that gathers multiple inputters."""

  def __init__(self, inputters):
    if not isinstance(inputters, list) or not inputters:
      raise ValueError("inputters must be a non empty list")
    dtype = inputters[0].dtype
    for inputter in inputters:
      if inputter.dtype != dtype:
        raise TypeError("All inputters must have the same dtype")
    super(MultiInputter, self).__init__(dtype=dtype)
    self.inputters = inputters

  @abc.abstractmethod
  def make_dataset(self, data_file):
    raise NotImplementedError()

  @abc.abstractmethod
  def get_dataset_size(self, data_file):
    raise NotImplementedError()

  def initialize(self, metadata):
    for inputter in self.inputters:
      inputter.initialize(metadata)

  def visualize(self, log_dir):
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        inputter.visualize(log_dir)

  @abc.abstractmethod
  def _get_serving_input(self):
    raise NotImplementedError()

  def transform(self, inputs, mode):
    transformed = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        transformed.append(inputter.transform(inputs[i], mode))
    return transformed


class ParallelInputter(MultiInputter):
  """An multi inputter that process parallel data."""

  def __init__(self, inputters, reducer=None):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of :class:`opennmt.inputters.inputter.Inputter`.
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all inputs. If
        set, parallel inputs are assumed to have the same length.
    """
    super(ParallelInputter, self).__init__(inputters)
    self.reducer = reducer

  def get_length(self, data):
    lengths = []
    for i, inputter in enumerate(self.inputters):
      sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
      lengths.append(inputter.get_length(sub_data))
    if self.reducer is None:
      return lengths
    else:
      return lengths[0]

  def make_dataset(self, data_file):
    if not isinstance(data_file, list) or len(data_file) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    datasets = [
        inputter.make_dataset(data)
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

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for i, inputter in enumerate(self.inputters):
      receiver_tensors, features = inputter._get_serving_input()  # pylint: disable=protected-access
      for key, value in six.iteritems(receiver_tensors):
        all_receiver_tensors["{}_{}".format(key, i)] = value
      for key, value in six.iteritems(features):
        all_features["inputter_{}_{}".format(i, key)] = value
    return all_receiver_tensors, all_features

  def _process(self, data):
    processed_data = {}
    for i, inputter in enumerate(self.inputters):
      sub_data = inputter._process(data[i])  # pylint: disable=protected-access
      for key, value in six.iteritems(sub_data):
        prefixed_key = "inputter_{}_{}".format(i, key)
        processed_data = self.set_data_field(
            processed_data,
            prefixed_key,
            value,
            volatile=key in inputter.volatile)
    return processed_data

  def _transform_data(self, data, mode):
    transformed = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
        transformed.append(inputter._transform_data(sub_data, mode))  # pylint: disable=protected-access
    if self.reducer is not None:
      transformed = self.reducer.reduce(transformed)
    return transformed

  def transform(self, inputs, mode):
    transformed = super(ParallelInputter, self).transform(inputs, mode)
    if self.reducer is not None:
      transformed = self.reducer.reduce(transformed)
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
    super(MixedInputter, self).__init__(inputters)
    self.reducer = reducer
    self.dropout = dropout

  def get_length(self, data):
    return self.inputters[0].get_length(data)

  def make_dataset(self, data_file):
    return self.inputters[0].make_dataset(data_file)

  def get_dataset_size(self, data_file):
    return self.inputters[0].get_dataset_size(data_file)

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for inputter in self.inputters:
      receiver_tensors, features = inputter._get_serving_input()  # pylint: disable=protected-access
      all_receiver_tensors.update(receiver_tensors)
      all_features.update(features)
    return all_receiver_tensors, all_features

  def _process(self, data):
    for inputter in self.inputters:
      data = inputter._process(data)  # pylint: disable=protected-access
      self.volatile |= inputter.volatile
    return data

  def _transform_data(self, data, mode):
    transformed = []
    for i, inputter in enumerate(self.inputters):
      with tf.variable_scope("inputter_{}".format(i)):
        transformed.append(inputter._transform_data(data, mode))  # pylint: disable=protected-access
    outputs = self.reducer.reduce(transformed)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs

  def transform(self, inputs, mode):
    transformed = super(MixedInputter, self).transform(inputs, mode)
    outputs = self.reducer.reduce(transformed)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs
