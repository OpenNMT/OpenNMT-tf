"""Defines common layers."""

import tensorflow as tf
import numpy as np

from opennmt.utils.misc import shape_list


def dropout(x, rate, training=None):
    """Simple dropout layer."""
    if not training or rate == 0:
        return x
    return tf.nn.dropout(x, rate)


def gelu(x):
    """Gaussian Error Linear Unit activation function described in
    https://arxiv.org/abs/1606.08415.
    """
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


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
        super().__init__(units, **kwargs)
        self.set_kernel(weight, transpose=transpose)

    def set_kernel(self, weight, transpose=False):
        """Use :obj:`weight` as the kernel weights matrix.

        Args:
          weight: The weight ot use.
          transpose: Whether :obj:`weight` should be transposed or not.

        Raises:
          ValueError: if the layer is already built.
        """
        if self.built:
            raise ValueError("The layer is already built")
        self.weight = weight
        self.transpose = transpose

    def add_weight(self, name, *args, **kwargs):
        if self.weight is not None and name == "kernel":
            return self.weight
        return super().add_weight(name, *args, **kwargs)

    def call(self, inputs):
        shape = shape_list(inputs)
        rank = len(shape)
        if rank > 2:
            inputs = tf.reshape(inputs, [-1, shape[-1]])
        outputs = tf.matmul(inputs, self.kernel, transpose_b=self.transpose)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        if rank > 2:
            outputs = tf.reshape(outputs, shape[:-1] + [self.units])
        return outputs

    def map_v1_weights(self, weights):
        m = [(self.kernel, weights["kernel"])]
        if self.use_bias:
            m.append((self.bias, weights["bias"]))
        return m


class LayerNorm(tf.keras.layers.LayerNormalization):
    """Layer normalization."""

    def map_v1_weights(self, weights):
        return [(self.beta, weights["beta"]), (self.gamma, weights["gamma"])]


class LayerWrapper(tf.keras.layers.Layer):
    """Layer wrapper for input/output normalization, input/output dropout and
    residual connection.
    """

    def __init__(
        self,
        layer,
        normalize_input=False,
        normalize_output=False,
        input_dropout=0,
        output_dropout=0,
        residual_connection=False,
        **kwargs
    ):
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
        super().__init__(**kwargs)
        self.layer = layer
        self.input_layer_norm = LayerNorm() if normalize_input else None
        self.output_layer_norm = LayerNorm() if normalize_output else None
        self.input_dropout = input_dropout
        self.output_dropout = output_dropout
        self.residual_connection = residual_connection

    def call(self, inputs, *args, **kwargs):
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
            "residual_connection": self.residual_connection,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        """Creates a layer wrapper from its configuration."""
        layer = tf.keras.layers.deserialize(config.pop("layer"))
        return cls(layer, **config)
