"""Define self-attention decoder."""

import tensorflow as tf
import opennmt.utils.transformer as transformer

from opennmt.encoders.encoder import create_position_embedding
from opennmt.decoders.decoder import Decoder
from opennmt.utils.reducer import SumReducer


class SelfAttentionDecoder(Decoder):
  """Decoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_heads=8,
               ffn_inner_dim=2048,
               dropout=0.1):
    """Initializes the parameters of the decoder.

    Args:
      num_layers: The number of layers.
      num_heads: The number of heads in the multi-head attention.
      ffn_inner_dim: The number of units of the inner linear transformation
        in the feed forward layer.
      dropout: The probability to drop units from the outputs.
    """
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.ffn_inner_dim = ffn_inner_dim
    self.dropout = dropout
    self.position_encoding_reducer = SumReducer()

  def decode(self,
             inputs,
             sequence_length,
             vocab_size,
             encoder_states=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None,
             return_logits=True):
    # TODO: implement positional encoding as described in the paper.
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
    # TODO: inherit from tf.contrib.seq2seq.Decoder to rely on dynamic_decode?

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
    tf.logging.warning("Beam search is not yet implemented, using greedy decoding instead.")

    return self.dynamic_decode(
      embeddings,
      start_tokens,
      end_token,
      vocab_size,
      encoder_states=encoder_states,
      maximum_iterations=maximum_iterations,
      mode=mode,
      memory=memory,
      memory_sequence_length=memory_sequence_length)
