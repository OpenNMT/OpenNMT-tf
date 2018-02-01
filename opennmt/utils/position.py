"""Define position encoder classes."""

import abc
import six

import tensorflow as tf

from opennmt.utils.reducer import SumReducer


def make_positions(sequence_length, maximum_length=None):
  """Builds a sequence of positions.

  The first position is 1 as the 0 index is reserved to padding positions.

  Args:
    sequence_length: The length of each sequence as a ``tf.Tensor`` of shape
      :math:`[B]`.
    maximum_length: Optional size of the returned time dimension. Otherwise it
      is the maximum of :obj:`sequence_length`.

  Returns:
    The sequence of positions as a ``tf.Tensor`` of shape :math:`[B, T]`.
  """
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)

  batch_size = tf.shape(sequence_length)[0]

  # Make 0 the position of padding.
  position = tf.range(maximum_length) + 1
  position = tf.tile(position, [batch_size])
  position = tf.reshape(position, [batch_size, -1])

  mask = tf.sequence_mask(
      sequence_length, maxlen=maximum_length, dtype=position.dtype)

  position = position * mask

  return position


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
      inputs: The inputs of shape :math:`[B, T, D]`.
      sequence_length: The length of each sequence of shape :math:`[B]`.
        If ``None``, sequences are assumed to have the same length.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D]` where :math:`D` depends on the
      :attr:`reducer`.
    """
    timesteps = tf.shape(inputs)[1]

    if sequence_length is None:
      batch_size = tf.shape(inputs)[0]
      sequence_length = tf.fill([batch_size], timesteps)

    input_dim = inputs.get_shape().as_list()[-1]

    with tf.variable_scope("position_encoding"):
      position_encoding = self.encode_sequence(
          sequence_length,
          input_dim,
          maximum_length=timesteps,
          dtype=inputs.dtype)
      return self.reducer.reduce([inputs, position_encoding])

  def apply_one(self, inputs, position):
    """Apply position encoding to one input.

    This is usually used during dynamic decoding.

    Args:
      inputs: The inputs of shape :math:`[B, 1, D]`.
      position: The position to encode.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, 1, D]` where :math:`D` depends on the
      :attr:`reducer`.
    """
    batch_size = tf.shape(inputs)[0]
    input_dim = inputs.get_shape().as_list()[-1]

    position = tf.tile([position], [batch_size])

    with tf.variable_scope("position_encoding"):
      position_encoding = self.encode(position, input_dim, dtype=inputs.dtype)
      position_encoding = tf.expand_dims(position_encoding, 1)
      return self.reducer.reduce([inputs, position_encoding])

  @abc.abstractmethod
  def encode(self, positions, depth, dtype=tf.float32):
    """Creates position encodings.

    Args:
      position: The positions to encode of shape :math:`[B, ...]`.
      depth: The encoding depth :math:`D`.
      dtype: The encoding type.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, ..., D]`.
    """
    raise NotImplementedError()

  def encode_sequence(self,
                      sequence_length,
                      depth,
                      maximum_length=None,
                      dtype=tf.float32):
    """Creates position encodings for sequences.

    Args:
      sequence_length: The length of each sequence of shape :math:`[B]`.
      depth: The encoding depth :math:`D`.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The encoding type.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    """
    positions = make_positions(sequence_length, maximum_length=maximum_length)
    return self.encode(positions, depth, dtype=dtype)


class PositionEmbedder(PositionEncoder):
  """Encodes position with a lookup table."""

  def __init__(self, maximum_position=128, reducer=SumReducer()):
    """Initializes the position encoder.

    Args:
      maximum_position: The maximum position to embed. Positions greater
        than this value will be set to :obj:`maximum_position`.
      reducer: A :class:`opennmt.utils.reducer.Reducer` to merge inputs and
        position encodings.
    """
    super(PositionEmbedder, self).__init__(reducer=reducer)
    self.maximum_position = maximum_position

  def encode(self, positions, depth, dtype=tf.float32):
    positions = tf.minimum(positions, self.maximum_position)
    embeddings = tf.get_variable(
        "w_embs", shape=[self.maximum_position + 1, depth], dtype=dtype)
    return tf.nn.embedding_lookup(embeddings, positions)
