"""Define self-attention decoder."""

import tensorflow as tf
import opennmt.utils.transformer as transformer

from opennmt.decoders.decoder import Decoder, get_embedding_fn
from opennmt.utils.position import PositionEmbedder
from opennmt.utils.beam_search import beam_search


class SelfAttentionDecoder(Decoder):
  """Decoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1,
               attention_dropout=0.1,
               relu_dropout=0.1,
               position_encoder=PositionEmbedder()):
    """Initializes the parameters of the decoder.

    Args:
      num_layers: The number of layers.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: A :class:`opennmt.utils.position.PositionEncoder` to
        apply on inputs or ``None``.
    """
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.relu_dropout = relu_dropout
    self.position_encoder = position_encoder

  def _self_attention_stack(self,
                            inputs,
                            sequence_length,
                            mode=tf.estimator.ModeKeys.TRAIN,
                            memory=None,
                            memory_sequence_length=None):
    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    inputs = tf.layers.dropout(
        inputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        with tf.variable_scope("masked_multi_head"):
          inputs_norm = transformer.norm(inputs)
          encoded = transformer.multi_head_attention(
              self.num_heads,
              inputs_norm,
              inputs_norm,
              inputs_norm,
              mode,
              values_length=sequence_length,
              mask_future=True,
              dropout=self.attention_dropout)
          encoded = transformer.drop_and_add(
              inputs,
              encoded,
              mode,
              dropout=self.dropout)

        with tf.variable_scope("multi_head"):
          values = memory if memory is not None else encoded
          keys = values

          context = transformer.multi_head_attention(
              self.num_heads,
              transformer.norm(encoded),
              keys,
              values,
              mode,
              values_length=memory_sequence_length,
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
             vocab_size,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    if sampling_probability is not None:
      raise ValueError("Scheduled sampling is not supported with SelfAttentionDecoder")

    outputs = self._self_attention_stack(
        inputs,
        sequence_length,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length)
    outputs = tf.layers.dense(outputs, vocab_size)

    return (outputs, None, sequence_length)

  def dynamic_decode(self,
                     embedding,
                     start_tokens,
                     end_token,
                     vocab_size,
                     initial_state=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None):
    batch_size = tf.shape(start_tokens)[0]
    finished = tf.tile([False], [batch_size])
    step = tf.constant(0)
    inputs = tf.expand_dims(start_tokens, 1)
    lengths = tf.zeros([batch_size], dtype=tf.int32)
    log_probs = tf.zeros([batch_size])

    embedding_fn = get_embedding_fn(embedding)

    def _condition(unused_step, finished, unused_inputs, unused_lengths, unused_log_probs):
      return tf.logical_not(tf.reduce_all(finished))

    def _body(step, finished, inputs, lengths, log_probs):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, tf.int32))

      # Decode inputs.
      outputs = self._self_attention_stack(
          embedding_fn(inputs),
          inputs_lengths,
          mode=mode,
          memory=memory,
          memory_sequence_length=memory_sequence_length)

      # Only sample the last timestep.
      last_output = tf.slice(outputs, [0, step, 0], [-1, 1, -1])
      logits = tf.layers.dense(
          last_output,
          vocab_size)
      probs = tf.nn.log_softmax(logits)
      sample_ids = tf.argmax(probs, axis=-1)

      # Accumulate log probabilities.
      sample_probs = tf.reduce_max(probs, axis=-1)
      masked_probs = tf.squeeze(sample_probs, -1) * (1.0 - tf.cast(finished, tf.float32))
      log_probs = tf.add(log_probs, masked_probs)

      next_inputs = tf.concat([inputs, tf.cast(sample_ids, tf.int32)], -1)
      next_lengths = inputs_lengths
      next_finished = tf.logical_or(
          finished,
          tf.equal(tf.squeeze(sample_ids, axis=[1]), end_token))
      step = step + 1

      if maximum_iterations is not None:
        next_finished = tf.logical_or(next_finished, step >= maximum_iterations)

      return step, next_finished, next_inputs, next_lengths, log_probs

    step, _, outputs, lengths, log_probs = tf.while_loop(
        _condition,
        _body,
        loop_vars=(step, finished, inputs, lengths, log_probs),
        shape_invariants=(
            tf.TensorShape([]),
            finished.get_shape(),
            tf.TensorShape([None, None]),
            lengths.get_shape(),
            log_probs.get_shape()
        ),
        parallel_iterations=1)

    outputs = tf.slice(outputs, [0, 1], [-1, -1]) # Ignore <s>.

    # Make shape consistent with beam search.
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    return (outputs, None, lengths, log_probs)


  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size,
                                initial_state=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None):
    if initial_state is not None:
      initial_state = tf.contrib.seq2seq.tile_batch(
          initial_state, multiplier=beam_width)
    if memory is not None:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
    if memory_sequence_length is not None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
          memory_sequence_length, multiplier=beam_width)

    embedding_fn = get_embedding_fn(embedding)

    def _symbols_to_logits_fn(symbols):
      batch_size = tf.shape(symbols)[0]
      step = tf.shape(symbols)[1]
      sequence_length = tf.fill([batch_size], step)
      outputs = self._self_attention_stack(
          embedding_fn(symbols),
          sequence_length,
          mode=mode,
          memory=memory,
          memory_sequence_length=memory_sequence_length)

      # Only sample the last timestep.
      last_output = tf.slice(outputs, [0, step - 1, 0], [-1, 1, -1])
      logits = tf.layers.dense(
          last_output,
          vocab_size)
      return logits

    outputs, log_probs = beam_search(
        _symbols_to_logits_fn,
        start_tokens,
        beam_width,
        maximum_iterations,
        vocab_size,
        length_penalty,
        eos_id=end_token)
    outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1]) # Ignore <s>.

    lengths = tf.not_equal(outputs, 0)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)

    return (outputs, None, lengths, log_probs)
