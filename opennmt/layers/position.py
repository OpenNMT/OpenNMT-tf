"""Define position encoder classes."""

import math
import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import SumReducer
from opennmt.utils import compat


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
class PositionEncoder(tf.keras.layers.Layer):
  """Base class for position encoders."""

  def __init__(self, reducer=SumReducer()):
    super(PositionEncoder, self).__init__()
    self.reducer = reducer

  def __call__(self, inputs, sequence_length=None, position=None):  # pylint: disable=arguments-differ
    """Apply position encoding to inputs.

    Args:
      inputs: The inputs of shape :math:`[B, T, D]`.
      sequence_length: The length of each sequence of shape :math:`[B]`.
        If ``None``, sequences are assumed to have the same length.
      position: If known, the position to encode (1-indexed).

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D]` where :math:`D` depends on the
      :attr:`reducer`.
    """
    if compat.is_tf2():
      return super(PositionEncoder, self).__call__(
          inputs, sequence_length=sequence_length, position=position)
    self._dtype = inputs.dtype
    # Build by default for backward compatibility.
    if not compat.reuse():
      self.build(inputs.shape)
    return self.call(
        inputs, sequence_length=sequence_length, position=position)

  def call(self, inputs, sequence_length=None, position=None):  # pylint: disable=arguments-differ
    _ = sequence_length

    batch_size = tf.shape(inputs)[0]
    timesteps = tf.shape(inputs)[1]
    input_dim = inputs.get_shape().as_list()[-1]

    if position is None:
      positions = tf.range(timesteps) + 1
    else:
      positions = [position]
    position_encoding = self.encode([positions], input_dim, dtype=inputs.dtype)
    position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
    return self.reducer([inputs, position_encoding])

  def apply(self, inputs, sequence_length=None):  # pylint: disable=arguments-differ
    """Shortcut for ``__call__``."""
    return self(inputs, sequence_length=sequence_length)

  def apply_one(self, inputs, position):
    """Shortcut for ``__call__``."""
    return self(inputs, position=position)

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
      reducer: A :class:`opennmt.layers.reducer.Reducer` to merge inputs and
        position encodings.
    """
    super(PositionEmbedder, self).__init__(reducer=reducer)
    self.maximum_position = maximum_position
    self.embedding = None

  def build(self, input_shape):
    shape = [self.maximum_position + 1, input_shape.as_list()[-1]]
    initializer = tf.keras.initializers.glorot_uniform()
    self.embedding = tf.Variable(
        initial_value=lambda: initializer(shape, dtype=self.dtype),
        name=compat.name_from_variable_scope("position_encoding/w_embs"))
    super(PositionEmbedder, self).build(input_shape)

  def encode(self, positions, depth, dtype=tf.float32):
    positions = tf.minimum(positions, self.maximum_position)
    return tf.nn.embedding_lookup(self.embedding, positions)


class SinusoidalPositionEncoder(PositionEncoder):
  """Encodes positions with sine waves as described in
  https://arxiv.org/abs/1706.03762.
  """

  def encode(self, positions, depth, dtype=tf.float32):
    if depth % 2 != 0:
      raise ValueError("SinusoidalPositionEncoder expects the depth to be divisble "
                       "by 2 but got %d" % depth)

    batch_size = tf.shape(positions)[0]
    positions = tf.cast(positions, tf.float32)

    log_timescale_increment = math.log(10000) / (depth / 2 - 1)
    inv_timescales = tf.exp(tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment)
    inv_timescales = tf.reshape(tf.tile(inv_timescales, [batch_size]), [batch_size, -1])
    scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
    encoding = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
    return tf.cast(encoding, dtype)
