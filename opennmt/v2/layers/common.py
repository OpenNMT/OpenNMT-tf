"""Defines common layers."""

import tensorflow as tf


class LayerNorm(tf.keras.layers.Layer):
  """Layer normalization."""

  def __init__(self, epsilon=1e-6, **kwargs):
    """Initializes this layer.

    Args:
      epsilon: The epsilon value to use.
      kwargs: Additional layer arguments.
    """
    super(LayerNorm, self).__init__(**kwargs)
    self.epsilon = epsilon

  def build(self, input_shape):
    """Creates the variables."""
    depth = input_shape[-1]
    self.bias = self.add_variable(
        "beta", [depth], initializer=tf.keras.initializers.Constant(0))
    self.scale = self.add_variable(
        "gamma", [depth], initializer=tf.keras.initializers.Constant(1))
    super(LayerNorm, self).build(input_shape)

  def call(self, x):  # pylint: disable=arguments-differ
    """Normalizes :obj:`x`."""
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.math.rsqrt(variance + self.epsilon)
    return norm_x * self.scale + self.bias
