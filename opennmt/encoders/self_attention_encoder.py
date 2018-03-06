"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.layers import transformer

from opennmt.encoders.encoder import Encoder
from opennmt.layers.position import SinusoidalPositionEncoder


class SelfAttentionEncoder(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               relu_dropout=0.1,
               position_encoder=SinusoidalPositionEncoder()):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    mask = transformer.build_sequence_mask(
        sequence_length,
        num_heads=self.num_heads,
        maximum_length=tf.shape(inputs)[1],
        dtype=inputs.dtype)

    state = ()

    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        with tf.variable_scope("multi_head"):
          context = transformer.multi_head_attention(
              self.num_heads,
              transformer.norm(inputs),
              None,
              mode,
              num_units=self.num_units,
              mask=mask,
              dropout=self.attention_dropout)
          context = transformer.drop_and_add(
              inputs,
              context,
              mode,
              dropout=self.dropout)

        with tf.variable_scope("ffn"):
          transformed = transformer.feed_forward(
              transformer.norm(context),
              self.ffn_inner_dim,
              mode,
              dropout=self.relu_dropout)
          transformed = transformer.drop_and_add(
              context,
              transformed,
              mode,
              dropout=self.dropout)

        inputs = transformed
        state += (tf.reduce_mean(inputs, axis=1),)

    outputs = transformer.norm(inputs)
    return (outputs, state, sequence_length)
