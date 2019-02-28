"""Defines common layers."""

import tensorflow as tf

from tensorflow.python.framework import function

from opennmt.utils import compat
from opennmt.utils.misc import shape_list


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Wraps :obj:`x` to convert its gradient to a tensor."""
  return x


def embedding_lookup(params, ids):
  """Wrapper around ``tf.nn.embedding_lookup``.

  This converts gradients of the embedding variable to tensors which allows
  to use of optimizers that don't support sparse gradients (e.g. Adafactor).

  Args:
    params: The embedding tensor.
    ids: The ids to lookup in :obj:`params`.

  Returns:
    A ``tf.Tensor``, the embeddings that correspond to :obj:`ids`.
  """
  params = convert_gradient_to_tensor(params)
  return tf.nn.embedding_lookup(params, ids)

def dropout(x, rate, training=None):
  """Simple dropout layer."""
  if not training or rate == 0:
    return x
  if compat.is_tf2():
    return tf.nn.dropout(x, rate)
  else:
    return tf.nn.dropout(x, 1.0 - rate)


class Dense(tf.keras.layers.Dense):
  """Small ``tf.keras.layers.Dense`` extension to possibly reuse an existing weight
  matrix.
  """

  def __init__(self, units, weight=None, transpose=False, **kwargs):
    """Initializes the layer.

    Args:
      unit: Positive integer, dimensionality of the output space.
      weight: The weight to reuse.
      transpose: Whether :obj:`weight` should be transposed or not.
      kwargs: Additional layers arguments.
    """
    super(Dense, self).__init__(units, **kwargs)
    self.weight = weight
    self.transpose = transpose

  def add_weight(self, name, *args, **kwargs):  # pylint: disable=arguments-differ
    if self.weight is not None and name == "kernel":
      return self.weight
    return super(Dense, self).add_weight(name, *args, **kwargs)

  def call(self, inputs):
    if self.weight is None:
      return super(Dense, self).call(inputs)
    shape = shape_list(inputs)
    rank = len(shape)
    if rank > 2:
      inputs = tf.reshape(inputs, [-1, shape[-1]])
    outputs = tf.matmul(inputs, self.kernel, transpose_b=self.transpose)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if rank > 2:
      outputs = tf.reshape(outputs, shape[:-1] + [self.units])
    return outputs


class LayerNorm(tf.keras.layers.Layer):
  """Layer normalization.

  Note:
    Requires TensorFlow 2.0.
  """

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


class LayerWrapper(tf.keras.layers.Layer):
  """Layer wrapper for input/output normalization, input/output dropout and
  residual connection.

  Note:
    Requires TensorFlow 2.0.
  """

  def __init__(self,
               layer,
               normalize_input=False,
               normalize_output=False,
               input_dropout=0,
               output_dropout=0,
               residual_connection=False,
               **kwargs):
    """Initializes the layer.

    Args:
      layer: The layer to wrap.
      normalize_input: Apply layer normalization on the input.
      normalize_output: Apply layer normalization on the output.
      input_dropout: The probability to drop units in the layer input.
      output_dropout: The probability to drop units in the layer output.
      residual_connection: Add the inputs to layer outputs (if their shape are
        compatible).
      kwargs: Additional layer arguments.
    """
    super(LayerWrapper, self).__init__(**kwargs)
    self.layer = layer
    self.input_layer_norm = LayerNorm(name="input_norm") if normalize_input else None
    self.output_layer_norm = LayerNorm(name="output_norm") if normalize_output else None
    self.input_dropout = input_dropout
    self.output_dropout = output_dropout
    self.residual_connection = residual_connection

  def call(self, inputs, *args, **kwargs):  # pylint: disable=arguments-differ
    """Runs the wrapper."""
    training = kwargs.get("training")
    x = inputs
    if self.input_layer_norm is not None:
      x = self.input_layer_norm(x)
    x = dropout(x, self.input_dropout, training=training)

    all_outputs = self.layer(x, *args, **kwargs)
    if isinstance(all_outputs, tuple):
      outputs = all_outputs[0]
      extra_outputs = list(all_outputs)[1:]
    else:
      outputs = all_outputs
      extra_outputs = None

    outputs = dropout(outputs, self.output_dropout, training=training)
    if self.residual_connection and outputs.shape[-1] == inputs.shape[-1]:
      outputs += inputs
    if self.output_layer_norm is not None:
      outputs = self.output_layer_norm(outputs)

    if extra_outputs:
      return tuple([outputs] + extra_outputs)
    return outputs

  # The wrapper should be serializable to be used in tf.keras.layers.Bidirectional.

  def get_config(self):
    """Returns the layer wrapper configuration."""
    config = {
        "layer": tf.keras.layers.serialize(self.layer),
        "normalize_input": self.input_layer_norm is not None,
        "normalize_output": self.output_layer_norm is not None,
        "input_dropout": self.input_dropout,
        "output_dropout": self.output_dropout,
        "residual_connection": self.residual_connection
    }
    base_config = super(LayerWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    """Creates a layer wrapper from its configuration."""
    layer = tf.keras.layers.deserialize(config.pop("layer"))
    return cls(layer, **config)
