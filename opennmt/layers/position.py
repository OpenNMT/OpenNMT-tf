"""Define position encoder classes."""

import math
import abc

import tensorflow as tf

from opennmt.layers.reducer import SumReducer


class PositionEncoder(tf.keras.layers.Layer):
    """Base class for position encoders."""

    def __init__(self, reducer=None, **kwargs):
        """Initializes the position encoder.

        Args:
          reducer: A :class:`opennmt.layers.Reducer` to merge inputs and position
            encodings. Defaults to :class:`opennmt.layers.SumReducer`.
          **kwargs: Additional layer keyword arguments.
        """
        super().__init__(**kwargs)
        if reducer is None:
            reducer = SumReducer(dtype=kwargs.get("dtype"))
        self.reducer = reducer

    def call(self, inputs, position=None):
        """Add position encodings to :obj:`inputs`.

        Args:
          inputs: The inputs to encode.
          position: The single position to encode, to use when this layer is called
            step by step.

        Returns:
          A ``tf.Tensor`` whose shape depends on the configured ``reducer``.
        """
        batch_size = tf.shape(inputs)[0]
        timesteps = tf.shape(inputs)[1]
        input_dim = inputs.shape[-1]
        positions = tf.range(timesteps) + 1 if position is None else [position]
        position_encoding = self._encode([positions], input_dim)
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])
        return self.reducer([inputs, position_encoding])

    @abc.abstractmethod
    def _encode(self, positions, depth):
        """Creates position encodings.

        Args:
          positions: The positions to encode of shape :math:`[B, ...]`.
          depth: The encoding depth :math:`D`.

        Returns:
          A ``tf.Tensor`` of shape :math:`[B, ..., D]`.
        """
        raise NotImplementedError()


class PositionEmbedder(PositionEncoder):
    """Encodes position with a lookup table."""

    def __init__(self, maximum_position=128, reducer=None, **kwargs):
        """Initializes the position encoder.

        Args:
          maximum_position: The maximum position to embed. Positions greater
            than this value will be set to :obj:`maximum_position`.
          reducer: A :class:`opennmt.layers.Reducer` to merge inputs and position
            encodings. Defaults to :class:`opennmt.layers.SumReducer`.
          **kwargs: Additional layer keyword arguments.
        """
        super().__init__(reducer=reducer, **kwargs)
        self.maximum_position = maximum_position
        self.embedding = None

    def build(self, input_shape):
        shape = [self.maximum_position + 1, input_shape[-1]]
        self.embedding = self.add_weight("position_embedding", shape)
        super().build(input_shape)

    def _encode(self, positions, depth):
        positions = tf.minimum(positions, self.maximum_position)
        return tf.nn.embedding_lookup(self.embedding, positions)


class SinusoidalPositionEncoder(PositionEncoder):
    """Encodes positions with sine waves as described in
    https://arxiv.org/abs/1706.03762.
    """

    def _encode(self, positions, depth):
        if depth % 2 != 0:
            raise ValueError(
                "SinusoidalPositionEncoder expects the depth to be divisble "
                "by 2 but got %d" % depth
            )

        batch_size = tf.shape(positions)[0]
        positions = tf.cast(positions, tf.float32)

        log_timescale_increment = math.log(10000) / (depth / 2 - 1)
        inv_timescales = tf.exp(
            tf.range(depth / 2, dtype=tf.float32) * -log_timescale_increment
        )
        inv_timescales = tf.reshape(
            tf.tile(inv_timescales, [batch_size]), [batch_size, depth // 2]
        )
        scaled_time = tf.expand_dims(positions, -1) * tf.expand_dims(inv_timescales, 1)
        encoding = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=2)
        return tf.cast(encoding, self.dtype)
