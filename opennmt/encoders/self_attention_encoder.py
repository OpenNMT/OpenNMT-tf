"""Define the self-attention encoder."""

import tensorflow as tf
import opennmt.utils.transformer as transformer

from opennmt.encoders.encoder import Encoder
from opennmt.utils.position import PositionEmbedder


class SelfAttentionEncoder(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               keep_layers_output=False,
               position_encoder=PositionEmbedder()):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      keep_layers_output: If ``True``, the memory of the encoder will contain
        the output of each layer. Otherwise, it will only contain the
        last layer output. This is ``True`` in the Transformer model.
      position_encoder: The :class:`opennmt.utils.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.position_encoder = position_encoder
    self.keep_layers_output = keep_layers_output

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    outputs = []
    state = ()

    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        with tf.variable_scope("multi_head"):
          context = transformer.multi_head_attention(
              self.num_heads,
              inputs,
              inputs,
              inputs,
              mode,
              values_length=sequence_length,
              dropout=self.dropout)
          context = transformer.add_and_norm(
              inputs,
              context,
              mode,
              dropout=self.dropout)

        with tf.variable_scope("ffn"):
          transformed = transformer.feed_forward(context, self.ffn_inner_dim)
          transformed = transformer.add_and_norm(
              context,
              transformed,
              mode,
              dropout=self.dropout)

        inputs = transformed
        state += (tf.reduce_mean(inputs, axis=1),)

        if self.keep_layers_output:
          outputs.append(inputs)

    return (inputs if not outputs else outputs, state, sequence_length)
