"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers import common


class SelfAttentionEncoder(Encoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        num_layers,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        maximum_relative_position=None,
        pre_norm=True,
        **kwargs
    ):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          ffn_dropout: The probability to drop units from the activation output in
            the feed forward layer.
          ffn_activation: The activation function to apply between the two linear
            transformations of the feed forward layer.
          position_encoder_class: The :class:`opennmt.layers.PositionEncoder`
            class to use for position encoding (or a callable that returns an
            instance).
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          pre_norm: If ``True``, layer normalization is applied before each
            sub-layer. Otherwise it is applied after.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.num_units = num_units
        self.dropout = dropout
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.layer_norm = common.LayerNorm() if pre_norm else None
        self.layers = [
            transformer.SelfAttentionEncoderLayer(
                num_units,
                num_heads,
                ffn_inner_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
            )
            for i in range(num_layers)
        ]

    def call(self, inputs, sequence_length=None, training=None):
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(inputs)
        inputs = common.dropout(inputs, self.dropout, training=training)
        mask = self.build_mask(inputs, sequence_length=sequence_length)
        for layer in self.layers:
            inputs = layer(inputs, mask=mask, training=training)
        outputs = self.layer_norm(inputs) if self.layer_norm is not None else inputs
        return outputs, None, sequence_length

    def map_v1_weights(self, weights):
        m = []
        m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
        for i, layer in enumerate(self.layers):
            m += layer.map_v1_weights(weights["layer_%d" % i])
        return m
