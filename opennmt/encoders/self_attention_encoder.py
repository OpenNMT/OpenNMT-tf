"""Define the self-attention encoder."""

import tensorflow as tf
import opennmt.utils.transformer as transformer

from opennmt.encoders.encoder import Encoder, create_position_embedding
from opennmt.utils.reducer import SumReducer


class SelfAttentionEncoder(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               keep_layers_output=False):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      keep_layers_output: If `True`, the states of the encoder will contain
        the complete output of each layers. Otherwise, it will contain the
        mean of these outputs. This is `True` in the Transformer model.
    """
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.keep_layers_output = keep_layers_output
    self.position_encoding_reducer = SumReducer()

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    # TODO: implements positional encoding as described in the paper.
    with tf.variable_scope("position_embedding"):
      input_dim = inputs.get_shape().as_list()[-1]
      position_embedding = create_position_embedding(
        input_dim,
        128,
        sequence_length)
      inputs = self.position_encoding_reducer.reduce(inputs, position_embedding)

    inputs = tf.layers.dropout(
      inputs,
      rate=self.dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

    states = []

    for l in range(self.num_layers):
      with tf.variable_scope("layer_" + str(l)):
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

        state = inputs
        if not self.keep_layers_output:
          state = tf.reduce_mean(state, axis=1)
        states.append(state)

    return (inputs, states, sequence_length)
