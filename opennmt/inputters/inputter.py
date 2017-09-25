"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import ConcatReducer
from opennmt.utils.misc import extract_prefixed_keys


@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  """Base class for inputters."""

  def __init__(self):
    self.resolved = False
    self.padded_shapes = {}
    self.process_hooks = []

  def add_process_hooks(self, hooks):
    """Adds processing hooks.

    Processing hooks are additional and model specific data processing
    functions applied after calling this inputter `process` function.

    Args:
      hooks: A list of callables with the signature `(inputter, data) -> data`.
    """
    self.process_hooks.extend(hooks)

  def set_data_field(self, data, key, value, padded_shape=[]):
    """Sets a data field.

    Args:
      data: The data dictionary.
      key: The value key.
      value: The value to assign.
      padded_shape: The padded shape of the value as given to
        `tf.contrib.data.Dataset.padded_batch`.

    Returns:
      The updated data dictionary.
    """
    data[key] = value
    self.padded_shapes[key] = padded_shape
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
    del self.padded_shapes[key]
    return data

  def get_length(self, data):
    """Returns the length of the input data, if defined."""
    return None

  @abc.abstractmethod
  def make_dataset(self, data_file):
    """Creates the dataset required by this inputter.

    Args:
      data_file: The data file.

    Returns:
      A `tf.contrib.data.Dataset`.
    """
    raise NotImplementedError()

  def get_serving_input_receiver(self):
    """Returns a serving input receiver for this inputter.

    Returns:
      A `tf.estimator.export.ServingInputReceiver`.
    """
    receiver_tensors, features = self._get_serving_input()
    # TODO: support batch input during preprocessing.
    for key, value in features.items():
      features[key] = tf.expand_dims(value, 0)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _get_serving_input(self):
    """Returns the input receiver for serving.

    Returns:
      A tuple `(receiver_tensors, features)` as described in
      `tf.estimator.export.ServingInputReceiver`.
    """
    raise NotImplementedError()

  def initialize(self, metadata):
    """Initializes the inputter within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    `TABLE_INITIALIZERS` collection.

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    pass

  def process(self, data):
    """Prepares raw data.

    See also `transform_data`.

    Args:
      data: The raw data.

    Returns:
      A dictionary of `tf.Tensor`s.
    """
    data = self._process(data)
    for hook in self.process_hooks:
      data = hook(self, data)
    return data

  def _process(self, data):
    """Prepares raw data (implementation).

    Subclasses should extend this function to prepare the raw value read
    from the dataset to something they can transform (e.g. processing a
    line of text to a sequence of ids).

    This base implementation makes sure the data is a dictionary so subclasses
    can populate it.

    Args:
      data: The raw data or a dictionary containing the `raw` key.

    Returns:
      A dictionary of `tf.Tensor`s.
    """
    if not isinstance(data, dict):
      data = self.set_data_field({}, "raw", data)
    elif "raw" not in data:
      raise ValueError("data must contain the raw dataset value")
    return data

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    pass

  def transform_data(self, data, mode, log_dir=None):
    """Transforms the processed data to an input.

    This is usually a simple forward of a `data` field to `transform`.

    See also `process`.

    Args:
      data: A dictionary of data fields.
      mode: A `tf.estimator.ModeKeys` mode.
      log_dir: The log directory. If set, visualization will be setup.

    Returns:
      The transformed input.
    """
    inputs = self._transform_data(data, mode)
    if log_dir:
      self.visualize(log_dir)
    return inputs

  def transform(self, inputs, mode, scope=None, reuse_next=None):
    """Transforms inputs.

    Args:
      inputs: A possible nested structure of `Tensor` depending on the inputter.
      mode: A `tf.estimator.ModeKeys` mode.
      scope: (optional) The variable scope to use.
      reuse: (optional) If `True`, reuse variables in this scope after the first call.

    Returns:
      The transformed input.
    """
    if scope is not None:
      reuse = reuse_next and self.resolved
      with tf.variable_scope(scope, reuse=reuse):
        outputs = self._transform(inputs, mode, reuse=reuse)
    else:
      outputs = self._transform(inputs, mode)
    self.resolved = True
    return outputs

  @abc.abstractmethod
  def _transform_data(self, data, mode):
    """Implementation of `transform_data`."""
    raise NotImplementedError()

  @abc.abstractmethod
  def _transform(self, inputs, mode, reuse=None):
    """Implementation of `transform`."""
    raise NotImplementedError()


@six.add_metaclass(abc.ABCMeta)
class MultiInputter(Inputter):
  """An inputter that gathers multiple inputters."""

  def __init__(self, inputters):
    super(MultiInputter, self).__init__()
    self.inputters = inputters

  @abc.abstractmethod
  def make_dataset(self, data_file):
    raise NotImplementedError()

  def initialize(self, metadata):
    for inputter in self.inputters:
      inputter.initialize(metadata)

  def visualize(self, log_dir):
    for i in range(len(self.inputters)):
      with tf.variable_scope("inputter_{}".format(i)):
        self.inputters[i].visualize(log_dir)

  @abc.abstractmethod
  def _get_serving_input(self):
    raise NotImplementedError()

  def _transform(self, inputs, mode, reuse=None):
    transformed = []
    for i in range(len(self.inputters)):
      with tf.variable_scope("inputter_{}".format(i), reuse=reuse):
        transformed.append(self.inputters[i]._transform(inputs[i], mode))  # pylint: disable=protected-access
    return transformed


class ParallelInputter(MultiInputter):
  """An multi inputter that process parallel data."""

  def __init__(self, inputters):
    """Initializes a parallel inputter.

    Args:
      inputters: A list of `onmt.inputters.Inputter`s.
    """
    super(ParallelInputter, self).__init__(inputters)

  def get_length(self, data):
    # Let the first inputter defines the input data length.
    sub_data = extract_prefixed_keys(data, "inputter_0_")
    return self.inputters[0].get_length(sub_data)

  def make_dataset(self, data_files):
    if not isinstance(data_files, list) or len(data_files) != len(self.inputters):
      raise ValueError("The number of data files must be the same as the number of inputters")
    datasets = [
        inputter.make_dataset(data_file)
        for inputter, data_file in zip(self.inputters, data_files)]
    return tf.contrib.data.Dataset.zip(tuple(datasets))

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for i in range(len(self.inputters)):
      receiver_tensors, features = self.inputters[i]._get_serving_input()  # pylint: disable=protected-access
      for key, value in receiver_tensors.items():
        all_receiver_tensors["{}_{}".format(key, i)] = value
      for key, value in features.items():
        all_features["inputter_{}_{}".format(i, key)] = value
    return all_receiver_tensors, all_features

  def _process(self, parallel_data):
    processed_data = {}
    for i in range(len(self.inputters)):
      data = self.inputters[i].process(parallel_data[i])
      for key, value in data.items():
        prefixed_key = "inputter_{}_{}".format(i, key)
        processed_data = self.set_data_field(
            processed_data,
            prefixed_key,
            value,
            padded_shape=self.inputters[i].padded_shapes[key])
    return processed_data

  def _transform_data(self, data, mode):
    transformed = []
    for i in range(len(self.inputters)):
      with tf.variable_scope("inputter_{}".format(i)):
        sub_data = extract_prefixed_keys(data, "inputter_{}_".format(i))
        transformed.append(self.inputters[i]._transform_data(sub_data, mode))  # pylint: disable=protected-access
    return transformed


class MixedInputter(MultiInputter):
  """An multi inputter that applies several transformation on the same data."""

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of `onmt.inputters.Inputter`s.
      reducer: A `onmt.utils.Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    super(MixedInputter, self).__init__(inputters)
    self.reducer = reducer
    self.dropout = dropout

  def get_length(self, data):
    return self.inputters[0].get_length(data)

  def make_dataset(self, data_file):
    return self.inputters[0].make_dataset(data_file)

  def _get_serving_input(self):
    all_receiver_tensors = {}
    all_features = {}
    for i in range(len(self.inputters)):
      receiver_tensors, features = self.inputters[i]._get_serving_input()  # pylint: disable=protected-access
      all_features.update(features)
      if i == 0:
        all_receiver_tensors = receiver_tensors
    return all_receiver_tensors, all_features

  def _process(self, data):
    for inputter in self.inputters:
      data = inputter.process(data)
      self.padded_shapes.update(inputter.padded_shapes)
    return data

  def _transform_data(self, data, mode):
    transformed = []
    for i in range(len(self.inputters)):
      with tf.variable_scope("inputter_{}".format(i)):
        transformed.append(self.inputters[i]._transform_data(data, mode))  # pylint: disable=protected-access
    return transformed

  def _transform(self, inputs, mode, reuse=None):
    transformed = super(MixedInputter, self)._transform(inputs, mode, reuse=None)
    outputs = self.reducer.reduce_all(transformed)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs
