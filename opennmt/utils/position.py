"""Define position encoder classes."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import SumReducer


@six.add_metaclass(abc.ABCMeta)
class PositionEncoder(object):
  """Base class for position encoders."""

  def __init__(self, reducer=SumReducer()):
    self.reducer = reducer

  def __call__(self, inputs, sequence_length=None):
    """Shortcut for `apply`."""
    return self.apply(inputs, sequence_length=sequence_length)

  def apply(self, inputs, sequence_length=None):
    """Apply position encoding to inputs.

    Args:
      inputs: The inputs to apply position encoding to.
      sequence_length: The length of each sequence of shape `[B]`.
        If `None`, sequences are assumed to have the same length.

    Returns:
      A `tf.Tensor` of shape `[B, T, D]` where `D` depends on the `reducer`.
    """
    if sequence_length is None:
      batch_size = tf.shape(inputs)[0]
      timesteps = tf.shape(inputs)[1]
      sequence_length = tf.fill([batch_size], timesteps)

    input_dim = inputs.get_shape().as_list()[-1]

    with tf.variable_scope("position_encoding"):
      position_encoding = self._encode(input_dim, sequence_length)
      return self.reducer.reduce(inputs, position_encoding)

  @abc.abstractmethod
  def _encode(self, input_dim, sequence_length):
    """Creates position encodings.

    Args:
      input_dim: The input dimension.
      sequence_length: The length of each sequence of shape `[B]`.

    Returns:
      A `tf.Tensor` of shape `[B, T, input_dim]`.
    """
    raise NotImplementedError()


class PositionEmbedder(PositionEncoder):
  """Encodes position with a lookup table."""

  def __init__(self, maximum_position=128, reducer=SumReducer()):
    """Initializes the position encoder.

    Args:
      maximum_position: The maximum position to embed. Positions greater
        than this value will be set to `maximum_position`.
      reducer: A `onmt.utils.Reducer` to merge inputs and position encodings.
    """
    super(PositionEmbedder, self).__init__(reducer=reducer)
    self.maximum_position = maximum_position

  def _encode(self, input_dim, sequence_length):
    maximum_length = tf.reduce_max(sequence_length)
    batch_size = tf.shape(sequence_length)[0]

    # Make 0 the position of padding.
    position = tf.range(maximum_length) + 1
    position = tf.tile(position, [batch_size])
    position = tf.reshape(position, [batch_size, -1])

    mask = tf.sequence_mask(sequence_length)
    mask = tf.cast(mask, tf.int32)

    position = position * mask
    position = tf.minimum(position, self.maximum_position)

    embeddings = tf.get_variable(
        "w_embs", shape=[self.maximum_position + 1, input_dim])

    return tf.nn.embedding_lookup(embeddings, position)
