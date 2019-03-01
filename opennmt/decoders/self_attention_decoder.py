"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.decoders import decoder
from opennmt.layers import common, transformer
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

  @property
  def support_multi_source(self):
    return True

  def _init_cache(self, batch_size, dtype=tf.float32, num_sources=1):
    cache = {}

    for l in range(self.num_layers):
      proj_cache_shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      layer_cache = {}
      layer_cache["memory"] = [
          {
              "memory_keys": tf.zeros(proj_cache_shape, dtype=dtype),
              "memory_values": tf.zeros(proj_cache_shape, dtype=dtype)
          } for _ in range(num_sources)]
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
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)

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

    if memory is not None and not tf.contrib.framework.nest.is_sequence(memory):
      memory = (memory,)
    if memory_sequence_length is not None:
      if not tf.contrib.framework.nest.is_sequence(memory_sequence_length):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = [
          transformer.build_sequence_mask(
              length, num_heads=self.num_heads, maximum_length=tf.shape(m)[1])
          for m, length in zip(memory, memory_sequence_length)]

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
            last_context = transformer.drop_and_add(
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
            last_context = transformer.drop_and_add(
                inputs, y, mode, dropout=self.dropout)

        if memory is not None:
          for i, (mem, mask) in enumerate(zip(memory, memory_mask)):
            memory_cache = layer_cache["memory"][i] if layer_cache is not None else None
            with tf.variable_scope("multi_head" if i == 0 else "multi_head_%d" % i):
              context, last_attention = transformer.multi_head_attention(
                  self.num_heads,
                  transformer.norm(last_context),
                  mem,
                  mode,
                  mask=mask,
                  cache=memory_cache,
                  dropout=self.attention_dropout,
                  return_attention=True)
              last_context = transformer.drop_and_add(
                  last_context,
                  context,
                  mode,
                  dropout=self.dropout)
              if i > 0:  # Do not return attention in case of multi source.
                last_attention = None

        with tf.variable_scope("ffn"):
          transformed = transformer.feed_forward(
              transformer.norm(last_context),
              self.ffn_inner_dim,
              mode,
              dropout=self.relu_dropout)
          transformed = transformer.drop_and_add(
              last_context,
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
    if memory is None:
      num_sources = 0
    elif tf.contrib.framework.nest.is_sequence(memory):
      num_sources = len(memory)
    else:
      num_sources = 1
    cache = self._init_cache(batch_size, dtype=dtype, num_sources=num_sources)
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
      if attention is not None:
        attention = tf.squeeze(attention, axis=1)
      return outputs, cache, attention
    return _fn, cache

class SelfAttentionDecoderV2(decoder.DecoderV2):
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
               num_sources=1,
               **kwargs):
    """Initializes the parameters of the decoder.

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
      num_sources: The number of source contexts expected by this decoder.
      **kwargs: Additional layer arguments.
    """
    super(SelfAttentionDecoderV2, self).__init__(num_sources=num_sources, **kwargs)
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.position_encoder = position_encoder
    self.layer_norm = common.LayerNorm(name="output_norm")
    self.layers = [
        _SelfAttentionDecoderLayer(
            self.num_units,
            self.num_heads,
            ffn_inner_dim,
            num_sources=num_sources,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            name="layer_%d" % i)
        for i in range(num_layers)]

  @property
  def minimum_sources(self):
    return 0

  @property
  def maximum_sources(self):
    return 1e6  # An arbitrary large number.

  def _run(self,
           inputs,
           sequence_length=None,
           cache=None,
           memory=None,
           memory_sequence_length=None,
           step=None,
           training=None):
    # Process inputs.
    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, position=step + 1 if step is not None else None)
    inputs = common.dropout(inputs, self.dropout, training=training)

    # Prepare query mask.
    mask = None
    if sequence_length is not None:
      mask = transformer.build_future_mask(
          sequence_length, maximum_length=tf.shape(inputs)[1])

    # Prepare memory mask.
    memory_mask = None
    if memory is not None:
      if not isinstance(memory, (list, tuple)):
        memory = (memory,)
    if memory_sequence_length is not None:
      if not isinstance(memory_sequence_length, (list, tuple)):
        memory_sequence_length = (memory_sequence_length,)
      memory_mask = []
      for mem, mem_length in zip(memory, memory_sequence_length):
        mem_mask = tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1], dtype=tf.float32)
        mem_mask = tf.expand_dims(mem_mask, 1)
        memory_mask.append(mem_mask)

    # Run each layer.
    new_cache = []
    for i, layer in enumerate(self.layers):
      inputs, layer_cache, attention = layer(
          inputs,
          mask=mask,
          memory=memory,
          memory_mask=memory_mask,
          cache=cache[i] if cache is not None else None,
          training=training)
      new_cache.append(layer_cache)
    outputs = self.layer_norm(inputs)
    return outputs, new_cache, attention

  def forward(self,
              inputs,
              sequence_length=None,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              training=None):
    _ = initial_state
    return self._run(
        inputs,
        sequence_length=sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=training)

  def step(self,
           inputs,
           timestep,
           state=None,
           memory=None,
           memory_sequence_length=None,
           training=None):
    inputs = tf.expand_dims(inputs, 1)
    outputs, state, attention = self._run(
        inputs,
        cache=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        step=timestep,
        training=training)
    outputs = tf.squeeze(outputs, axis=1)
    if attention is not None:
      attention = tf.squeeze(attention, axis=1)
    return outputs, state, attention

  def _get_initial_state(self, batch_size, dtype, initial_state=None):
    # The decoder state contains the keys and values projections of the previous timesteps.
    _ = initial_state
    cache = []
    for _ in self.layers:
      shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
      self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
      memory_kv = [
          (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
          for _ in range(self.num_sources)]
      cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
    return cache


class _SelfAttentionDecoderLayer(tf.keras.layers.Layer):
  """Implements one self-attention decoding layer."""

  def __init__(self,
               num_units,
               num_heads,
               ffn_inner_dim,
               num_sources=1,
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
      num_sources: The number of source contexts.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      ffn_dropout: The probability to drop units from the activation output in
        the feed forward layer.
      ffn_activation: The activation function to apply between the two linear
        transformations of the feed forward layer.
      **kwargs: Additional layer arguments.
    """
    super(_SelfAttentionDecoderLayer, self).__init__(**kwargs)
    self.self_attention = transformer.MultiHeadAttention(
        num_heads,
        num_units,
        dropout=attention_dropout,
        name="masked_multi_head_attention")
    self.self_attention = transformer.TransformerLayerWrapper(
        self.self_attention, dropout, name="sub_layer_0")
    self.attention = []
    for i in range(num_sources):
      attention = transformer.MultiHeadAttention(
          num_heads,
          num_units,
          dropout=attention_dropout,
          return_attention=num_sources == 1,
          name="multi_head_attention")
      attention = transformer.TransformerLayerWrapper(
          attention, dropout, name="sub_layer_%d" % (i + 1))
      self.attention.append(attention)
    self.ffn = transformer.FeedForwardNetwork(
        ffn_inner_dim,
        num_units,
        dropout=ffn_dropout,
        activation=ffn_activation,
        name="feed_forward")
    self.ffn = transformer.TransformerLayerWrapper(
        self.ffn, dropout, name="sub_layer_%d" % (num_sources + 1))

  # pylint: disable=arguments-differ
  def call(self,
           inputs,
           mask=None,
           memory=None,
           memory_mask=None,
           cache=None,
           training=None):
    """Runs the decoder layer."""
    if cache is None:
      cache = {}

    outputs, self_kv = self.self_attention(
        inputs,
        mask=mask,
        cache=cache.get("self_kv"),
        training=training)

    attention = None
    memory_kv = []
    if memory is not None:
      memory_cache = cache.get("memory_kv")
      if memory_cache is None:
        memory_cache = [None] * len(self.attention)
      for layer, mem, mem_mask, mem_cache in zip(
          self.attention, memory, memory_mask, memory_cache):
        result = layer(
            outputs,
            memory=mem,
            mask=mem_mask,
            cache=mem_cache,
            training=training)
        if len(result) == 3:
          outputs, memory_kv_i, attention = result
          attention = attention[:, 0]  # Use the first head for the attention vector.
        else:
          outputs, memory_kv_i = result
        memory_kv.append(memory_kv_i)

    outputs = self.ffn(outputs, training=training)
    cache = dict(self_kv=self_kv, memory_kv=memory_kv)
    return outputs, cache, attention
