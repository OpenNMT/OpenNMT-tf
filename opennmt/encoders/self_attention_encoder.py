"""Define the self-attention encoder."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder, create_position_embedding
from opennmt.utils.reducer import SumReducer
from opennmt.utils.attention import multi_head_attention


def transformer_ffn(x, inner_dim):
  """Implements the Transformer's "Feed Forward" layer.

  ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2

  Args:
    x: The input.
    inner_dim: The number of units of the inner linear transformation.

  Returns:
    The transformed input.
  """
  input_dim = x.get_shape().as_list()[-1]

  with tf.variable_scope("ffn"):
    inner = tf.layers.dense(
      inputs=x,
      units=inner_dim,
      activation=tf.nn.relu)
    outer = tf.layers.dense(
      inputs=inner,
      units=input_dim)

    return outer

def transformer_add_and_norm(inputs,
                             outputs,
                             mode,
                             dropout=0.1):
  """Implements the Transformer's "Add & Norm" layer.

  Args:
    inputs: The input of the previous layer.
    outputs: The output of the previous layer.
    mode: A `tf.estimator.ModeKeys` mode.
    dropout: The probability to drop units in `outputs`.

  Returns:
    The residual and normalized output.
  """
  outputs = tf.contrib.layers.dropout(
    outputs,
    keep_prob=1.0 - dropout,
    is_training=mode == tf.estimator.ModeKeys.TRAIN)
  outputs += inputs
  outputs = tf.contrib.layers.layer_norm(outputs)
  return outputs


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

    states = []

    for l in range(self.num_layers):
      with tf.variable_scope("layer_" + str(l)):
        context = multi_head_attention(
          self.num_heads,
          inputs,
          inputs,
          inputs,
          mode,
          values_length=sequence_length,
          dropout=self.dropout)
        context = transformer_add_and_norm(
          inputs,
          context,
          mode,
          dropout=self.dropout)

        transformed = transformer_ffn(context, self.ffn_inner_dim)
        transformed = transformer_add_and_norm(
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
