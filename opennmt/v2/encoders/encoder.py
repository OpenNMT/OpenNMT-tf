"""Base class for encoders and generic multi encoders."""

import abc
import six

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Encoder(tf.keras.layers.Layer):
  """Base class for encoders."""

  def call(self, inputs, sequence_length=None, training=None):
    """Encodes an input sequence.

    Args:
      inputs: The inputs to encode of shape :math:`[B, T, ...]`.
      sequence_length: The length of each input with shape :math:`[B]`.
      training: Run in training mode.

    Returns:
      A tuple ``(outputs, state, sequence_length)``.
    """
    return self.encode(inputs, sequence_length=sequence_length, training=training)

  def build_mask(self, inputs, sequence_length=None, dtype=tf.bool):
    """Builds a boolean mask for :obj:`inputs`."""
    if sequence_length is None:
      return None
    mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1], dtype=dtype)
    mask = tf.expand_dims(mask, -1)
    return mask

  @abc.abstractmethod
  def encode(self, inputs, sequence_length=None, training=None):
    """Encodes an input sequence.

    Args:
      inputs: The inputs to encode of shape :math:`[B, T, ...]`.
      sequence_length: The length of each input with shape :math:`[B]`.
      training: Run in training mode.

    Returns:
      A tuple ``(outputs, state, sequence_length)``.
    """
    raise NotImplementedError()
