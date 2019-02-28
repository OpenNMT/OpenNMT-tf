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
    super(SelfAttentionEncoder, self).__init__()
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
      inputs = self.position_encoder(inputs)

    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    mask = transformer.build_sequence_mask(
        sequence_length,
        num_heads=self.num_heads,
        maximum_length=tf.shape(inputs)[1])

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


class SelfAttentionEncoderV2(Encoder):
  """Encoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.

  Note:
    TensorFlow 2.0 version.
  """

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               position_encoder=SinusoidalPositionEncoder(),
               **kwargs):
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
      position_encoder: The :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs.
    """
    super(SelfAttentionEncoderV2, self).__init__(**kwargs)
    self.num_units = num_units
    self.dropout = dropout
    self.position_encoder = position_encoder
    self.layer_norm = common.LayerNorm()
    self.layers = [
        _SelfAttentionEncoderLayer(
            num_units,
            num_heads,
            ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            name="layer_%d" % i)
        for i in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=None):
    """Encodes :obj:`inputs`."""
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs)
    inputs = common.dropout(inputs, self.dropout, training=training)
    mask = None
    if sequence_length is not None:
      mask = tf.sequence_mask(sequence_length, maxlen=tf.shape(inputs)[1], dtype=tf.float32)
      mask = tf.expand_dims(mask, 1)
    for layer in self.layers:
      inputs = layer(inputs, mask=mask, training=training)
    outputs = self.layer_norm(inputs)
    return outputs, None, sequence_length


class _SelfAttentionEncoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention encoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               attention_dropout=0.1,
               ffn_dropout=0.1,
               ffn_activation=tf.nn.relu,
               **kwargs):
    """Initializes the layer.

    Args:
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
      kwargs: Additional layer arguments.
    """
    super(_SelfAttentionEncoderLayer, self).__init__(**kwargs)
    self.self_attention = transformer.MultiHeadAttention(
        num_heads, num_units, dropout=attention_dropout, name="multi_head_attention")
    self.self_attention = transformer.TransformerLayerWrapper(
        self.self_attention, dropout, name="sub_layer_0")
    self.ffn = transformer.FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation,
        name="feed_forward")
    self.ffn = transformer.TransformerLayerWrapper(
        self.ffn, dropout, name="sub_layer_1")

  def call(self, x, mask=None, training=None):  # pylint: disable=arguments-differ
    """Runs the encoder layer."""
    y, _ = self.self_attention(x, mask=mask, training=training)
    y = self.ffn(y, training=training)
    return y
