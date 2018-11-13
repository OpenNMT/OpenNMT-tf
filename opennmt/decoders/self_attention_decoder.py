"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.decoders import decoder
from opennmt.layers import transformer
from opennmt.layers.position import SinusoidalPositionEncoder


class SelfAttentionDecoder(decoder.Decoder):
  """Decoder using self-attention as described in
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
               position_encoder=SinusoidalPositionEncoder(),
               self_attention_type="scaled_dot"):
    """Initializes the parameters of the decoder.

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
      position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
        apply on inputs or ``None``.
      self_attention_type: Type of self attention, "scaled_dot" or "average" (case
        insensitive).

    Raises:
      ValueError: if :obj:`self_attention_type` is invalid.
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder
    self.self_attention_type = self_attention_type.lower()
    if self.self_attention_type not in ("scaled_dot", "average"):
      raise ValueError("invalid attention type %s" % self.self_attention_type)
    if self.self_attention_type == "average":
      tf.logging.warning("Support for average attention network is experimental "
                         "and may change in future versions.")

  @property
  def output_size(self):
    """Returns the decoder output size."""
    return self.num_units

  @property
  def support_alignment_history(self):
    return True

  def _init_cache(self, batch_size, dtype=tf.float32):
    cache = {}

    for l in range(self.num_layers):
      proj_cache_shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      layer_cache = {
          "memory_keys": tf.zeros(proj_cache_shape, dtype=dtype),
          "memory_values": tf.zeros(proj_cache_shape, dtype=dtype),
      }
      if self.self_attention_type == "scaled_dot":
        layer_cache["self_keys"] = tf.zeros(proj_cache_shape, dtype=dtype)
        layer_cache["self_values"] = tf.zeros(proj_cache_shape, dtype=dtype)
      elif self.self_attention_type == "average":
        layer_cache["prev_g"] = tf.zeros([batch_size, 1, self.num_units], dtype=dtype)
      cache["layer_{}".format(l)] = layer_cache

    return cache

  def _self_attention_stack(self,
                            inputs,
                            sequence_length=None,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            cache=None,
                            memory=None,
                            memory_sequence_length=None,
                            step=None):
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      if step is None:
        inputs = self.position_encoder(inputs, sequence_length=sequence_length)
      else:
        inputs = self.position_encoder.apply_one(inputs, step + 1)

    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder_mask = None
    memory_mask = None
    last_attention = None

    if self.self_attention_type == "scaled_dot":
      if sequence_length is not None:
        decoder_mask = transformer.build_future_mask(
            sequence_length,
            num_heads=self.num_heads,
            maximum_length=tf.shape(inputs)[1])
    elif self.self_attention_type == "average":
      if cache is None:
        if sequence_length is None:
          sequence_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
        decoder_mask = transformer.cumulative_average_mask(
            sequence_length, maximum_length=tf.shape(inputs)[1], dtype=inputs.dtype)

    if memory is not None and memory_sequence_length is not None:
      memory_mask = transformer.build_sequence_mask(
          memory_sequence_length,
          num_heads=self.num_heads,
          maximum_length=tf.shape(memory)[1])

    for l in range(self.num_layers):
      layer_name = "layer_{}".format(l)
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        if self.self_attention_type == "scaled_dot":
          with tf.variable_scope("masked_multi_head"):
            encoded = transformer.multi_head_attention(
                self.num_heads,
                transformer.norm(inputs),
                None,
                mode,
                num_units=self.num_units,
                mask=decoder_mask,
                cache=layer_cache,
                dropout=self.attention_dropout)
            encoded = transformer.drop_and_add(
                inputs,
                encoded,
                mode,
                dropout=self.dropout)
        elif self.self_attention_type == "average":
          with tf.variable_scope("average_attention"):
            # Cumulative average.
            x = transformer.norm(inputs)
            y = transformer.cumulative_average(
                x, decoder_mask if cache is None else step, cache=layer_cache)
            # FFN.
            y = transformer.feed_forward(
                y, self.ffn_inner_dim, mode, dropout=self.relu_dropout)
            # Gating layer.
            z = tf.layers.dense(tf.concat([x, y], -1), self.num_units * 2)
            i, f = tf.split(z, 2, axis=-1)
            y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
            encoded = transformer.drop_and_add(
                inputs, y, mode, dropout=self.dropout)

        if memory is not None:
          with tf.variable_scope("multi_head"):
            context, last_attention = transformer.multi_head_attention(
                self.num_heads,
                transformer.norm(encoded),
                memory,
                mode,
                mask=memory_mask,
                cache=layer_cache,
                dropout=self.attention_dropout,
                return_attention=True)
            context = transformer.drop_and_add(
                encoded,
                context,
                mode,
                dropout=self.dropout)
        else:
          context = encoded

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

    if last_attention is not None:
      # The first head of the last layer is returned.
      first_head_attention = last_attention[:, 0]
    else:
      first_head_attention = None

    outputs = transformer.norm(inputs)
    return outputs, first_head_attention

  def decode_from_inputs(self,
                         inputs,
                         sequence_length,
                         initial_state=None,
                         mode=tf.estimator.ModeKeys.TRAIN,
                         memory=None,
                         memory_sequence_length=None):
    outputs, attention = self._self_attention_stack(
        inputs,
        sequence_length=sequence_length,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length)
    return outputs, None, attention

  def step_fn(self,
              mode,
              batch_size,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              dtype=tf.float32):
    cache = self._init_cache(batch_size, dtype=dtype)
    def _fn(step, inputs, cache, mode):
      inputs = tf.expand_dims(inputs, 1)
      outputs, attention = self._self_attention_stack(
          inputs,
          mode=mode,
          cache=cache,
          memory=memory,
          memory_sequence_length=memory_sequence_length,
          step=step)
      outputs = tf.squeeze(outputs, axis=1)
      attention = tf.squeeze(attention, axis=1)
      return outputs, cache, attention
    return _fn, cache
