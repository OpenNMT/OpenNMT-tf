"""Defines common layers."""

import tensorflow as tf

from opennmt.utils.misc import function_args


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


class LayerWrapper(tf.keras.layers.Wrapper):
  """Layer wrapper for input/output normalization and residual connections."""

  def __init__(self,
               layer,
               normalize_input=True,
               normalize_output=False,
               residual_connections=True,
               dropout=0.1,
               **kwargs):
    """Initializes the layer.

    Args:
      layer: The layer to wrap.
      normalize_input: Apply layer normalization on the input.
      normalize_output: Apply layer normalization on the output.
      residual_connections: Add the inputs to layer outputs (if their shape are
        compatible).
      dropout: The probability to drop units in the layer output.
      kwargs: Additional layer arguments.
    """
    super(LayerWrapper, self).__init__(layer, **kwargs)
    self.input_layer_norm = LayerNorm() if normalize_input else None
    self.output_layer_norm = LayerNorm() if normalize_output else None
    self.residual_connections = residual_connections
    self.dropout = dropout

  def call(self, inputs, training=None, **kwargs):  # pylint: disable=arguments-differ
    """Runs the wrapper."""
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm(x)
    if "training" in function_args(self.layer.call):
      kwargs["training"] = training
    all_outputs = self.layer(x, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None
    if training:
      outputs = tf.nn.dropout(outputs, rate=self.dropout)
    if self.residual_connections and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm(outputs)
    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs
