"""Define convolution-based encoders."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder
from opennmt.layers import common
from opennmt.layers import position


class ConvEncoder(Encoder):
    """An encoder that applies a convolution over the input sequence
    as described in https://arxiv.org/abs/1611.02344.
    """

    def __init__(
        self,
        num_layers_a,
        num_layers_c,
        num_units,
        kernel_size=3,
        dropout=0.3,
        position_encoder_class=position.PositionEmbedder,
    ):
        """Initializes the parameters of the encoder.

        Args:
          num_layers_a: The number of layers in CNN-a.
          num_layers_c: The number of layers in CNN-c.
          num_units: The number of output filters.
          kernel_size: The kernel size.
          dropout: The probability to drop units from the inputs.
          position_encoder_class: The :class:`opennmt.layers.PositionEncoder`
            class to use for position encoding (or a callable that returns an
            instance).
        """
        super().__init__()
        self.dropout = dropout
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.cnn_a = [
            tf.keras.layers.Conv1D(num_units, kernel_size, padding="same")
            for _ in range(num_layers_a)
        ]
        self.cnn_c = [
            tf.keras.layers.Conv1D(num_units, kernel_size, padding="same")
            for _ in range(num_layers_c)
        ]

    def call(self, inputs, sequence_length=None, training=None):
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)
        inputs = common.dropout(inputs, self.dropout, training=training)

        cnn_a = _cnn_stack(self.cnn_a, inputs)
        cnn_c = _cnn_stack(self.cnn_c, inputs)

        outputs = cnn_a
        state = tf.reduce_mean(cnn_c, axis=1)
        return (outputs, state, sequence_length)


def _cnn_stack(layers, inputs):
    next_input = inputs

    for i, layer in enumerate(layers):
        outputs = layer(next_input)
        # Add residual connections past the first layer.
        if i > 0:
            outputs += next_input
        next_input = tf.tanh(outputs)

    return next_input
