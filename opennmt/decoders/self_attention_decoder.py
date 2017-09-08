"""Define self-attention decoder."""

import tensorflow as tf
import opennmt.utils.transformer as transformer

from opennmt.decoders.decoder import Decoder
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
               position_encoder=PositionEmbedder()):
    """Initializes the parameters of the decoder.

    Args:
      num_layers: The number of layers.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
      position_encoder: The `PositionEncoder` to apply on inputs or `None`.
    """
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.position_encoder = position_encoder

  def decode(self,
             inputs,
             sequence_length,
             vocab_size,
             encoder_states=None,
             scheduled_sampling_probability=0.0,
             embeddings=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None,
             return_logits=True):
    if scheduled_sampling_probability > 0:
      raise ValueError("Scheduled sampling is not supported with SelfAttentionDecoder")

    if self.position_encoder is not None:
      inputs = self.position_encoder(inputs, sequence_length=sequence_length)

    inputs = tf.layers.dropout(
      inputs,
      rate=self.dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

    for l in range(self.num_layers):
      with tf.variable_scope("layer_" + str(l)):
        with tf.variable_scope("masked_multi_head"):
          encoded = transformer.multi_head_attention(
            self.num_heads,
            inputs,
            inputs,
            inputs,
            mode,
            values_length=sequence_length,
            mask_future=True,
            dropout=self.dropout)
          encoded = transformer.add_and_norm(
            inputs,
            encoded,
            mode,
            dropout=self.dropout)

        with tf.variable_scope("multi_head"):
          if tf.contrib.framework.nest.is_sequence(memory):
            if l >= len(memory):
              raise ValueError("""If the encoder memory is a sequence,
                               it must contain one memory per decoder layer""")
            values = memory[l]
          else:
            values = memory
          keys = values

          context = transformer.multi_head_attention(
            self.num_heads,
            encoded,
            keys,
            values,
            mode,
            values_length=memory_sequence_length,
            dropout=self.dropout)
          context = transformer.add_and_norm(
            encoded,
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

    outputs = inputs

    if return_logits:
      outputs = tf.layers.dense(
        outputs,
        vocab_size)

    return (outputs, None, None)

  def dynamic_decode(self,
                     embeddings,
                     start_tokens,
                     end_token,
                     vocab_size,
                     encoder_states=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.TRAIN,
                     memory=None,
                     memory_sequence_length=None):
    batch_size = tf.shape(start_tokens)[0]
    finished = tf.tile([False], [batch_size])
    step = tf.constant(0)
    inputs = tf.expand_dims(start_tokens, 1)
    lengths = tf.zeros([batch_size], dtype=tf.int32)

    def condition(step, finished, inputs, lengths):
      return tf.logical_not(tf.reduce_all(finished))

    def body(step, finished, inputs, lengths):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, tf.int32))

      # Decode inputs.
      outputs, _, _ = self.decode(
        embeddings(inputs),
        inputs_lengths,
        vocab_size,
        encoder_states=encoder_states,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        return_logits=False)

      # Only sample the last timestep.
      last_output = tf.slice(outputs, [0, step, 0], [-1, 1, -1])
      logits = tf.layers.dense(
        last_output,
        vocab_size)
      probs = tf.nn.softmax(logits)
      sample_ids = tf.argmax(probs, axis=-1)

      next_inputs = tf.concat([inputs, tf.cast(sample_ids, tf.int32)], -1)
      next_lengths = inputs_lengths
      next_finished = tf.logical_or(
        finished,
        tf.equal(tf.squeeze(sample_ids, axis=[1]), end_token))
      step = step + 1

      if maximum_iterations is not None:
        next_finished = tf.logical_or(next_finished, step >= maximum_iterations)

      return step, next_finished, next_inputs, next_lengths

    res = tf.while_loop(
      condition,
      body,
      loop_vars=(step, finished, inputs, lengths),
      shape_invariants=(
        tf.TensorShape([]),
        finished.get_shape(),
        tf.TensorShape([None, None]),
        lengths.get_shape()
      ),
      parallel_iterations=32)

    step = res[0]
    lengths = res[3]
    outputs = tf.slice(res[2], [0, 1], [-1, -1]) # Ignore <s>.

    # Make shape consistent with beam search.
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)

    return (outputs, None, lengths)


  def dynamic_decode_and_search(self,
                                embeddings,
                                start_tokens,
                                end_token,
                                vocab_size,
                                encoder_states=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.TRAIN,
                                memory=None,
                                memory_sequence_length=None):
    if not encoder_states is None:
      encoder_states = tf.contrib.seq2seq.tile_batch(
        encoder_states, multiplier=beam_width)
    if not memory is None:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    if not memory_sequence_length is None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
        memory_sequence_length, multiplier=beam_width)

    def symbols_to_logits_fn(symbols):
      batch_size = tf.shape(symbols)[0]
      step = tf.shape(symbols)[1]
      sequence_length = tf.fill([batch_size], step)
      outputs, _, _ = self.decode(
        embeddings(symbols),
        sequence_length,
        vocab_size,
        encoder_states=encoder_states,
        mode=mode,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        return_logits=False)

      # Only sample the last timestep.
      last_output = tf.slice(outputs, [0, step - 1, 0], [-1, 1, -1])
      logits = tf.layers.dense(
        last_output,
        vocab_size)
      return logits

    outputs, _ = beam_search(
      symbols_to_logits_fn,
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

    return (outputs, None, lengths)
