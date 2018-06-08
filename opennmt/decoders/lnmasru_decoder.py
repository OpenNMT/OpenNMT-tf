"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.layers import transformer
from opennmt.layers import sr_nmt as srnmt
from opennmt.utils import beam_search

from opennmt.decoders.decoder import Decoder, get_embedding_fn, build_output_layer

class LNMASRUDecoder(Decoder):
  """Decoder using self-attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_layers,
               num_units=512,
               num_heads=8,
               dropout=0.1,
               attention_dropout=0.1,
               bridge = None):
    """Initializes the parameters of the decoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in the multi-head attention.
      dropout: The probability to drop units from the outputs.
      attention_dropout: The probability to drop units from the attention.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.num_heads = num_heads
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.bridge = bridge

  def _build_memory_mask(self, memory, memory_sequence_length=None):
    if memory_sequence_length is None:
      return None
    else:
      return transformer.build_sequence_mask(
          memory_sequence_length,
          num_heads=None,
          maximum_length=tf.shape(memory)[1],
          dtype=memory.dtype)

  def _symbols_to_logits_fn(self, embedding, vocab_size, mode, memory, memory_sequence_length, output_layer=None, dtype=None):
    embedding_fn = get_embedding_fn(embedding)
    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype)

    def _impl(ids, step, states):
      inputs = embedding_fn(ids[:, -1:])
      outputs, state = self._lnmasru_stack(
               inputs,
               states,
               memory=memory,
               mode=mode,
               memory_sequence_length=memory_sequence_length)
      outputs = outputs[:, -1:, :]
      logits = output_layer(outputs)
      return logits, state

    return _impl


  def _lnmasru_stack(self,
                    inputs,
                    state,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    memory=None,
                    memory_sequence_length=None):

    memory_mask = None

    if memory is not None:
      if memory_sequence_length is not None:
        memory_mask = self._build_memory_mask(
            memory, memory_sequence_length=memory_sequence_length)

    new_state = ()
    inputs = tf.transpose(inputs, [1, 0, 2])
    for l in range(self.num_layers):
      layer_name = "layer_{}".format(l)
      with tf.variable_scope(layer_name):
        inputs, n_state = srnmt.sr_decoder_unit(inputs,
                                                state[l], memory,
                                                memory_mask, mode,
                                                self.num_units, self.dropout,
                                                self.attention_dropout,
                                                self.num_heads)
        new_state += (n_state, )
    outputs = tf.transpose(inputs, [1, 0, 2])
    return outputs, new_state


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
      raise ValueError("Scheduled sampling is not supported with SR-NMT Decoder")

    initial_state = tuple([tf.zeros(tf.shape(initial_state),
                                    name='initial_state_f_{}'.format(l)) for l in range(self.num_layers)])

    outputs, new_state = self._lnmasru_stack(
        inputs,
        initial_state,
        mode=mode,
        memory=memory,
        memory_sequence_length=None)

    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=inputs.dtype)
    logits = output_layer(outputs)

    return logits, new_state, sequence_length

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

    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, memory=memory, memory_sequence_length=memory_sequence_length,
        output_layer=output_layer, dtype=dtype or memory.dtype)

    initial_state = tuple([tf.zeros(tf.shape(initial_state),
                                    name='initial_state_f_{}'.format(l)) for l in range(self.num_layers)])

    def _condition(unused_step, unused_state, finished, unused_inputs,
                   unused_lengths, unused_log_probs):
      return tf.logical_not(tf.reduce_all(finished))

    def _body(step, state, finished, inputs, lengths, log_probs):
      inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

      logits, next_state = symbols_to_logits_fn(inputs, step, state)
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

      return step, next_state, next_finished, next_inputs, next_lengths, log_probs

    step, state, _, outputs, lengths, log_probs= tf.while_loop(
        _condition,
        _body,
        loop_vars=(step, initial_state, finished, inputs, lengths, log_probs),
        shape_invariants=(
            tf.TensorShape([]),
            initial_state.get_shape(),
            finished.get_shape(),
            tf.TensorShape([None, None]),
            lengths.get_shape(),
            log_probs.get_shape() #,
        ),
        parallel_iterations=1)

    outputs = tf.slice(outputs, [0, 1], [-1, -1]) # Ignore <s>.

    # Make shape consistent with beam search.
    outputs = tf.expand_dims(outputs, 1)
    lengths = tf.expand_dims(lengths, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    if return_alignment_history:
      return (outputs, state, lengths, log_probs, None)
    return (outputs, state, lengths, log_probs)


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

    if memory is not None:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
    if memory_sequence_length is not None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
          memory_sequence_length, multiplier=beam_width)

    symbols_to_logits_fn = self._symbols_to_logits_fn(
        embedding, vocab_size, mode, memory=memory, memory_sequence_length=memory_sequence_length,
        output_layer=output_layer, dtype=dtype or memory.dtype)

    initial_state = tuple([tf.zeros(tf.shape(initial_state),
                                    name='initial_state_f_{}'.format(l)) for l in range(self.num_layers)])

    outputs, log_probs = beam_search.beam_search(
        symbols_to_logits_fn,
        start_tokens,
        beam_width,
        maximum_iterations,
        vocab_size,
        length_penalty,
        states=initial_state,
        eos_id=end_token)
    outputs = tf.slice(outputs, [0, 0, 1], [-1, -1, -1]) # Ignore <s>.

    lengths = tf.not_equal(outputs, 0)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)

    if return_alignment_history:
      return (outputs, None, lengths, log_probs, None)
    return (outputs, None, lengths, log_probs)


