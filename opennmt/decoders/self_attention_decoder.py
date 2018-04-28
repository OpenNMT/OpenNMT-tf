"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.layers import transformer
from opennmt.utils import beam_search

from opennmt.decoders.decoder import Decoder, get_embedding_fn, build_output_layer
from opennmt.layers.position import SinusoidalPositionEncoder


class SelfAttentionDecoder(Decoder):
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
               position_encoder=SinusoidalPositionEncoder()):
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
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder

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
    cache = {
        "memory": memory,
        "memory_mask": self._build_memory_mask(
            memory, memory_sequence_length=memory_sequence_length)
    }

    batch_size = tf.shape(memory)[0]
    depth = memory.get_shape().as_list()[-1]

    for l in range(self.num_layers):
      cache["layer_{}".format(l)] = {
          "self_keys": tf.zeros([batch_size, 0, depth]),
          "self_values": tf.zeros([batch_size, 0, depth]),
          "memory_keys": tf.zeros([batch_size, 0, depth]),
          "memory_values": tf.zeros([batch_size, 0, depth]),
      }

    return cache

  def _symbols_to_logits_fn(self, embedding, vocab_size, mode, output_layer=None, dtype=None):
    embedding_fn = get_embedding_fn(embedding)
    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype)

    def _impl(ids, step, cache):
      inputs = embedding_fn(ids[:, -1:])
      inputs *= self.num_units**0.5
      inputs = self.position_encoder.apply_one(inputs, step + 1)
      outputs = self._self_attention_stack(
          inputs,
          mode=mode,
          cache=cache,
          memory=cache["memory"],
          memory_sequence_length=None)
      outputs = outputs[:, -1:, :]
      logits = output_layer(outputs)
      return logits, cache

    return _impl

  def _self_attention_stack(self,
                            inputs,
                            sequence_length=None,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            cache=None,
                            memory=None,
                            memory_sequence_length=None):
    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    decoder_mask = None
    memory_mask = None

    if sequence_length is not None:
      decoder_mask = transformer.build_future_mask(
          sequence_length,
          num_heads=self.num_heads,
          maximum_length=tf.shape(inputs)[1],
          dtype=inputs.dtype)
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

        if memory is not None:
          with tf.variable_scope("multi_head"):
            context = transformer.multi_head_attention(
                self.num_heads,
                transformer.norm(encoded),
                memory,
                mode,
                mask=memory_mask,
                cache=layer_cache,
                dropout=self.attention_dropout)
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

    outputs = transformer.norm(inputs)
    return outputs

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
             memory_sequence_length=None):
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported with SelfAttentionDecoder")

    inputs *= self.num_units**0.5
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    outputs = self._self_attention_stack(
        inputs,
        sequence_length=sequence_length,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=inputs.dtype)
    logits = output_layer(outputs)

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
    batch_size = tf.shape(start_tokens)[0]
    finished = tf.tile([False], [batch_size])
    step = tf.constant(0)
    inputs = tf.expand_dims(start_tokens, 1)
    lengths = tf.zeros([batch_size], dtype=tf.int32)
    log_probs = tf.zeros([batch_size])
    cache = self._init_cache(memory, memory_sequence_length=memory_sequence_length)

    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, output_layer=output_layer, dtype=dtype or memory.dtype)

    def _condition(unused_step, finished, unused_inputs,
                   unused_lengths, unused_log_probs, unused_cache):
      return tf.logical_not(tf.reduce_all(finished))

    def _body(step, finished, inputs, lengths, log_probs, cache):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

      logits, cache = symbols_to_logits_fn(inputs, step, cache)
      probs = tf.nn.log_softmax(logits)
      sample_ids = tf.argmax(probs, axis=-1)

      # Accumulate log probabilities.
      sample_probs = tf.reduce_max(probs, axis=-1)
      masked_probs = tf.squeeze(sample_probs, -1) * (1.0 - tf.cast(finished, sample_probs.dtype))
      log_probs = tf.add(log_probs, masked_probs)

      next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
      next_lengths = inputs_lengths
      next_finished = tf.logical_or(
          finished,
          tf.equal(tf.squeeze(sample_ids, axis=[1]), end_token))
      step = step + 1

      if maximum_iterations is not None:
        next_finished = tf.logical_or(next_finished, step >= maximum_iterations)

      return step, next_finished, next_inputs, next_lengths, log_probs, cache

    step, _, outputs, lengths, log_probs, _ = tf.while_loop(
        _condition,
        _body,
        loop_vars=(step, finished, inputs, lengths, log_probs, cache),
        shape_invariants=(
            tf.TensorShape([]),
            finished.get_shape(),
            tf.TensorShape([None, None]),
            lengths.get_shape(),
            log_probs.get_shape(),
            tf.contrib.framework.nest.map_structure(beam_search.get_state_shape_invariants, cache)
        ),
        parallel_iterations=1)

    outputs = tf.slice(outputs, [0, 1], [-1, -1]) # Ignore <s>.

    # Make shape consistent with beam search.
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    if return_alignment_history:
      return (outputs, None, lengths, log_probs, None)
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

    outputs, log_probs = beam_search.beam_search(
        symbols_to_logits_fn,
        start_tokens,
        beam_width,
        maximum_iterations,
        vocab_size,
        length_penalty,
        states=cache,
        eos_id=end_token)
    outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1]) # Ignore <s>.

    lengths = tf.not_equal(outputs, 0)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)

    if return_alignment_history:
      return (outputs, None, lengths, log_probs, None)
    return (outputs, None, lengths, log_probs)
