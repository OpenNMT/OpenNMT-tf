"""Define generic embedders."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import ConcatReducer


@six.add_metaclass(abc.ABCMeta)
class Embedder(object):
  """Base class for embedders."""

  def __init__(self, name=None):
    self.name = name
    self.resolved = False
    self.padded_shapes = {}

  def set_name(self, name):
    self.name = name

  def _prefix_key(self, key):
    """Prefix the data `key` with the embedder's name (if defined)."""
    if not self.name is None:
      key = self.name + "_" + key
    return key

  def set_data_field(self, data, key, value, padded_shape=[]):
    """Sets a prefixed data field.

    Args:
      data: The data dictionary.
      key: The key to prefix with the embedder's name.
      value: The value to assign.
      padded_shape: The padded shape of the value as given to
        `tf.contrib.data.Dataset.padded_batch`.

    Returns:
      The updated data dictionary.
    """
    key = self._prefix_key(key)
    data[key] = value
    self.padded_shapes[key] = padded_shape
    return data

  def remove_data_field(self, data, key):
    """Remove a prefixed data field.

    Args:
      data: The data dictionary.
      key: The key to prefix with the embedder's name.

    Returns:
      The updated data dictionary.
    """
    key = self._prefix_key(key)
    del data[key]
    del self.padded_shapes[key]
    return data

  def get_data_field(self, data, key):
    """Gets a prefixed data field.

    Args:
      data: The data dictionary.
      key: The key to prefix with the embedder's name.

    Returns:
      The data field.
    """
    key = self._prefix_key(key)
    return data[key]

  def has_data_field(self, data, key):
    """Checks key existence.

    Args:
      data: The data dictionary.
      key: The key to prefix with the embedder's name.

    Returns:
      `True` if the data dictionary contains the prefixed key.
    """
    return self._prefix_key(key) in data

  def init(self):
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
    elif not self.has_data_field(data, "raw"):
      raise ValueError("data must contain the raw dataset value")
    return data

  def visualize(self, log_dir):
    """Optionally visualize embeddings."""
    pass

  @abc.abstractmethod
  def embed_from_data(self, data, mode):
    """Embeds inputs from the processed data.

    This is usually a simple forward of a `data` field to `embed`.
    See also `process`.

    Args:
      data: A dictionary of data fields.
      mode: A `tf.estimator.ModeKeys` mode.

    Returns:
      The embedding.
    """
    raise NotImplementedError()

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
        outputs = self._embed(inputs, mode)
    else:
      outputs = self._embed(inputs, mode)
    self.resolved = True
    return outputs

  @abc.abstractmethod
  def _embed(self, inputs, mode):
    """Implementation of `embed`."""
    raise NotImplementedError()


class MixedEmbedder(Embedder):
  """An embedder that mixes several embedders."""

  def __init__(self, embedders, reducer=ConcatReducer(), name=None):
    """Initializes a mixed embedder.

    Args:
      embedders: A list of `Embedder`.
      reducer: A `Reducer` to merge all embeddings.
    """
    super(MixedEmbedder, self).__init__(name=name)
    self.embedders = embedders
    self.reducer = reducer

    if name:
      self.set_name(name)

  def set_name(self, name):
    self.name = name
    for embedder in self.embedders:
      embedder.set_name(name)

  def process(self, data):
    for embedder in self.embedders:
      data = embedder.process(data)
    return data

  def embed_from_data(self, data, mode):
    embs = []
    for embedder in self.embedders:
      embs.append(embedder.embed_from_data(data, mode))
    return self.reducer.reduce_all(embs)

  def _embed(self, inputs, mode):
    embs = []
    for embedder, elem in zip(self.embedders, inputs):
      # TODO: call to each _embed method should be in different scopes.
      embs.append(embedder._embed(elem, mode))
    return self.reducer.reducal_all(embs)
