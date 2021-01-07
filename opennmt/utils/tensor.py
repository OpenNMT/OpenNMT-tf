"""Various tensor manipulation functions."""

import tensorflow as tf


def roll_sequence(tensor, offsets):
    """Shifts sequences by an offset.

    Args:
      tensor: A ``tf.Tensor`` of shape :math:`[B, T, ...]`.
      offsets : The offset of each sequence of shape :math:`[B]`.

    Returns:
      A ``tf.Tensor`` with the same shape as :obj:`tensor` and with sequences
      shifted by :obj:`offsets`.
    """
    batch_size = tf.shape(tensor)[0]
    time = tf.shape(tensor)[1]
    cols, rows = tf.meshgrid(tf.range(time), tf.range(batch_size))
    cols -= tf.expand_dims(offsets, 1)
    cols %= time
    indices = tf.stack([rows, cols], axis=-1)
    return tf.gather_nd(tensor, indices)
