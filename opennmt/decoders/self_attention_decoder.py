"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.decoders import decoder
from opennmt.layers import transformer
from opennmt.utils import beam_search
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

  def _build_memory_mask(self, memory, memory_sequence_length=None):
    if memory_sequence_length is None:
      return None
    else:
      return transformer.build_sequence_mask(
          memory_sequence_length,
          num_heads=self.num_heads,
          maximum_length=tf.shape(memory)[1],
          dtype=memory.dtype)

  def _init_cache(self, memory, memory_sequence_length=None):
    batch_size = tf.shape(memory)[0]
    memory_time = tf.shape(memory)[1]
    depth = memory.get_shape().as_list()[-1]

    cache = {
        "attn": tf.zeros([batch_size, 0, memory_time]),
        "memory": memory,
        "memory_mask": self._build_memory_mask(
            memory, memory_sequence_length=memory_sequence_length)
    }

    for l in range(self.num_layers):
      proj_cache_shape = [batch_size, self.num_heads, 0, depth // self.num_heads]
      layer_cache = {
          "memory_keys": tf.zeros(proj_cache_shape),
          "memory_values": tf.zeros(proj_cache_shape),
      }
      if self.self_attention_type == "scaled_dot":
        layer_cache["self_keys"] = tf.zeros(proj_cache_shape)
        layer_cache["self_values"] = tf.zeros(proj_cache_shape)
      elif self.self_attention_type == "average":
        layer_cache["prev_g"] = tf.zeros([batch_size, 1, depth])
      cache["layer_{}".format(l)] = layer_cache

    return cache

  def _symbols_to_logits_fn(self, embedding, vocab_size, mode, output_layer=None, dtype=None):
    embedding_fn = decoder.get_embedding_fn(embedding)
    if output_layer is None:
      output_layer = decoder.build_output_layer(self.num_units, vocab_size, dtype=dtype)

    def _impl(ids, step, cache):
      inputs = embedding_fn(ids[:, -1:])
      inputs *= self.num_units**0.5
      inputs = self.position_encoder.apply_one(inputs, step + 1)
      outputs, _ = self._self_attention_stack(
          inputs,
          mode=mode,
          cache=cache,
          memory=cache["memory"],
          memory_sequence_length=None,
          step=step)
      logits = output_layer(outputs)
      return logits, cache

    return _impl

  def _self_attention_stack(self,
                            inputs,
                            sequence_length=None,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            cache=None,
                            memory=None,
                            memory_sequence_length=None,
                            step=None):
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
            maximum_length=tf.shape(inputs)[1],
            dtype=inputs.dtype)
    elif self.self_attention_type == "average":
      if cache is None:
        if sequence_length is None:
          sequence_length = tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
        decoder_mask = transformer.cumulative_average_mask(
            sequence_length, maximum_length=tf.shape(inputs)[1], dtype=inputs.dtype)

    if memory is not None:
      if cache is not None:
        memory_mask = cache["memory_mask"]
      elif memory_sequence_length is not None:
        memory_mask = self._build_memory_mask(
            memory, memory_sequence_length=memory_sequence_length)

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

    # The first head of the last layer is returned.
    first_head_attention = last_attention[:, 0]
    if cache is not None and "attn" in cache:
      cache["attn"] = tf.concat([cache["attn"], first_head_attention], 1)

    outputs = transformer.norm(inputs)
    return outputs, first_head_attention

  def decode(self,
             inputs,
             sequence_length,
             vocab_size=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None,
             return_alignment_history=False):
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported with SelfAttentionDecoder")

    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    outputs, attention = self._self_attention_stack(
        inputs,
        sequence_length=sequence_length,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    if output_layer is None:
      output_layer = decoder.build_output_layer(self.num_units, vocab_size, dtype=inputs.dtype)
    logits = output_layer(outputs)

    if return_alignment_history:
      return (logits, None, sequence_length, attention)
    return (logits, None, sequence_length)

  def dynamic_decode(self,
                     embedding,
                     start_tokens,
                     end_token,
                     vocab_size=None,
                     initial_state=None,
                     output_layer=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None,
                     dtype=None,
                     return_alignment_history=False):
    cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)
    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype or memory.dtype)

    outputs, lengths, log_probs, cache = decoder.greedy_decode(
        symbols_to_logits_fn,
        start_tokens,
        end_token,
        decode_length=maximum_iterations,
        state=cache,
        return_state=True)
    outputs = tf.slice(outputs, [0, 1], [-1, -1]) # Ignore <s>.

    # Make shape consistent with beam search.
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    if return_alignment_history:
      attention = tf.expand_dims(cache["attn"], 1)
      return (outputs, None, lengths, log_probs, attention)
    return (outputs, None, lengths, log_probs)


  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size=None,
                                initial_state=None,
                                output_layer=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None,
                                dtype=None,
                                return_alignment_history=False):
    cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)
    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype or memory.dtype)

    outputs, log_probs, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        start_tokens,
        beam_width,
        maximum_iterations,
        vocab_size,
        length_penalty,
        states=cache,
        eos_id=end_token,
        return_states=True)
    outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1]) # Ignore <s>.

    lengths = tf.not_equal(outputs, 0)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)

    if return_alignment_history:
      return (outputs, None, lengths, log_probs, cache["attn"])
    return (outputs, None, lengths, log_probs)
