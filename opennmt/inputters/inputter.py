"""Define generic inputters."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import ConcatReducer


@six.add_metaclass(abc.ABCMeta)
class Inputter(object):
  """Base class for inputters."""

  def __init__(self):
    self.resolved = False
    self.padded_shapes = {}

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

  def make_dataset(self, data_file, resources):
    """Creates the dataset required by this inputter.

    Args:
      data_file: The data file.
      resources: A dictionary containing additional resources set
        by the user.

    Returns:
      A `tf.contrib.data.Dataset`.
    """
    self._initialize(resources)
    dataset = self._make_dataset(data_file)
    dataset = dataset.map(self.process)
    return dataset

  @abc.abstractmethod
  def _make_dataset(self, data_file):
    raise NotImplementedError()

  def _initialize(self, resources):
    """Initializes the inputter within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    `TABLE_INITIALIZERS` collection.

    Args:
      resources: A dictionary containing additional resources set
        by the user.
    """
    pass

  def process(self, data):
    """Transforms input from the dataset.

    Subclasses should extend this function to transform the raw value read
    from the dataset to an input they can consume. See also `transform_data`.

    This base implementation makes sure the data is a dictionary so subclasses
    can populate it.

    Args:
      data: The raw data or a dictionary containing the `raw` key.

    Returns:
      A dictionary.
    """
    if not isinstance(data, dict):
      data = self.set_data_field({}, "raw", data)
    elif not "raw" in data:
      raise ValueError("data must contain the raw dataset value")
    return data

  def visualize(self, log_dir):
    """Visualizes the transformation, usually embeddings.

    Args:
      log_dir: The active log directory.
    """
    pass

  def transform_data(self, data, mode, log_dir=None):
    """Transform the processed data to an input.

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
    """Transform inputs.

    Args:
      inputs: A possible nested structure of `Tensor` depending on the inputter.
      mode: A `tf.estimator.ModeKeys` mode.
      scope: (optional) The variable scope to use.
      reuse: (optional) If `True`, reuse variables in this scope after the first call.

    Returns:
      The transformed input.
    """
    if not scope is None:
      reuse = reuse_next and self.resolved
      with tf.variable_scope(scope, reuse=reuse):
        outputs = self._transform(inputs, mode, reuse=reuse)
    else:
      outputs = self._transform(inputs, mode)
    self.resolved = True
    return outputs

  @abc.abstractmethod
  def _transform_data(self, data, mode):
    raise NotImplementedError()

  @abc.abstractmethod
  def _transform(self, inputs, mode, reuse=None):
    """Implementation of `transform`."""
    raise NotImplementedError()


class MixedInputter(Inputter):
  """An inputter that mixes several inputters."""

  def __init__(self,
               inputters,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed inputter.

    Args:
      inputters: A list of `Inputter`.
      reducer: A `Reducer` to merge all inputs.
      dropout: The probability to drop units in the merged inputs.
    """
    super(MixedInputter, self).__init__()
    self.inputters = inputters
    self.reducer = reducer
    self.dropout = dropout

  def _make_dataset(self, data_file):
    return self.inputters[0]._make_dataset(data_file)

  def _initialize(self):
    for inputter in self.inputters:
      inputter._initialize()

  def process(self, data):
    for inputter in self.inputters:
      data = inputter.process(data)
      self.padded_shapes.update(inputter.padded_shapes)
    return data

  def visualize(self, log_dir):
    index = 0
    for inputter in self.inputters:
      with tf.variable_scope("inputter_" + str(index)):
        inputter.visualize(log_dir)
      index += 1

  def _transform_data(self, data, mode):
    embs = []
    index = 0
    for inputter in self.inputters:
      with tf.variable_scope("inputter_" + str(index)):
        embs.append(inputter._transform_data(data, mode)) # pylint: disable=protected-access
      index += 1
    return self.reducer.reduce_all(embs)

  def _transform(self, inputs, mode, reuse=None):
    embs = []
    index = 0
    for inputter, elem in zip(self.inputters, inputs):
      with tf.variable_scope("inputter_" + str(index), reuse=reuse):
        embs.append(inputter._transform(elem, mode)) # pylint: disable=protected-access
      index += 1
    outputs = self.reducer.reducal_all(embs)
    outputs = tf.layers.dropout(
      outputs,
      rate=self.dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs
