"""Define generic embedders."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import ConcatReducer


@six.add_metaclass(abc.ABCMeta)
class Embedder(object):
  """Base class for embedders."""

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
    """Remove a data field.

    Args:
      data: The data dictionary.
      key: The value key.

    Returns:
      The updated data dictionary.
    """
    del data[key]
    del self.padded_shapes[key]
    return data

  def make_dataset(self, data_file):
    """Creates the dataset required by this embedder.

    Args:
      data_file: The data file.

    Returns:
      A `tf.contrib.data.Dataset`.
    """
    self._initialize()
    dataset = self._make_dataset(data_file)
    dataset = dataset.map(self.process)
    return dataset

  @abc.abstractmethod
  def _make_dataset(self, data_file):
    raise NotImplementedError()

  def _initialize(self):
    """Initializes the embedder within the current graph.

    For example, one can create lookup tables in this method
    for their initializer to be added to the current graph
    `TABLE_INITIALIZERS` collection.
    """
    pass

  def process(self, data):
    """Transforms input from the dataset.

    Subclasses should extend this function to transform the raw value read
    from the dataset to an input they can consume. See also `embed_from_data`.

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
    """Optionally visualize embeddings."""
    pass

  def embed_from_data(self, data, mode, log_dir=None):
    """Embeds inputs from the processed data.

    This is usually a simple forward of a `data` field to `embed`.
    See also `process`.

    Args:
      data: A dictionary of data fields.
      mode: A `tf.estimator.ModeKeys` mode.
      log_dir: The log directory. If set, embeddings visualization
        will be setup.

    Returns:
      The embedding.
    """
    embed = self._embed_from_data(data, mode)
    if log_dir:
      self.visualize(log_dir)
    return embed

  def embed(self, inputs, mode, scope=None, reuse_next=None):
    """Embeds inputs.

    Args:
      inputs: A possible nested structure of `Tensor` depending on the embedder.
      mode: A `tf.estimator.ModeKeys` mode.
      scope: (optional) The variable scope to use.
      reuse: (optional) If `True`, reuse variables in this scope after the first call.

    Returns:
      The embedding.
    """
    if not scope is None:
      reuse = reuse_next and self.resolved
      with tf.variable_scope(scope, reuse=reuse):
        outputs = self._embed(inputs, mode, reuse=reuse)
    else:
      outputs = self._embed(inputs, mode)
    self.resolved = True
    return outputs

  @abc.abstractmethod
  def _embed_from_data(self, data, mode):
    raise NotImplementedError()

  @abc.abstractmethod
  def _embed(self, inputs, mode, reuse=None):
    """Implementation of `embed`."""
    raise NotImplementedError()


class MixedEmbedder(Embedder):
  """An embedder that mixes several embedders."""

  def __init__(self,
               embedders,
               reducer=ConcatReducer(),
               dropout=0.0):
    """Initializes a mixed embedder.

    Args:
      embedders: A list of `Embedder`.
      reducer: A `Reducer` to merge all embeddings.
      dropout: The probability to drop units in the merged embedding.
    """
    super(MixedEmbedder, self).__init__()
    self.embedders = embedders
    self.reducer = reducer
    self.dropout = dropout

  def _make_dataset(self, data_file):
    return self.embedders[0]._make_dataset(data_file)

  def _initialize(self):
    for embedder in self.embedders:
      embedder._initialize()

  def process(self, data):
    for embedder in self.embedders:
      data = embedder.process(data)
      self.padded_shapes.update(embedder.padded_shapes)
    return data

  def visualize(self, log_dir):
    for embedder in self.embedders:
      embedder.visualize(log_dir)

  def _embed_from_data(self, data, mode):
    embs = []
    index = 0
    for embedder in self.embedders:
      with tf.variable_scope("embedder_" + str(index)):
        embs.append(embedder._embed_from_data(data, mode)) # pylint: disable=protected-access
      index += 1
    return self.reducer.reduce_all(embs)

  def _embed(self, inputs, mode, reuse=None):
    embs = []
    index = 0
    for embedder, elem in zip(self.embedders, inputs):
      with tf.variable_scope("embedder_" + str(index), reuse=reuse):
        embs.append(embedder._embed(elem, mode)) # pylint: disable=protected-access
      index += 1
    outputs = self.reducer.reducal_all(embs)
    outputs = tf.layers.dropout(
      outputs,
      rate=self.dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs
