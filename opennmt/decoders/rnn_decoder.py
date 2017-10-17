"""Define RNN-based decoders."""

import tensorflow as tf

from opennmt.decoders.decoder import Decoder, logits_to_cum_log_probs
from opennmt.utils.cell import build_cell


class RNNDecoder(Decoder):
  """A basic RNN-based decoder."""

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A `onmt.utils.Bridge` to pass the encoder state to the decoder.
      cell_class: The inner cell class.
      dropout: The probability to drop units in each layer output.
      residual_connections: If `True`, each layer input will be added to its output.
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.bridge = bridge
    self.cell_class = cell_class
    self.dropout = dropout
    self.residual_connections = residual_connections

  def _init_state(self, zero_state, initial_state=None):
    if initial_state is None:
      return zero_state
    elif self.bridge is None:
      raise ValueError("A bridge must be configured when passing encoder state")
    else:
      return self.bridge(initial_state, zero_state)

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None):
    _ = memory
    _ = memory_sequence_length

    cell = build_cell(
        self.num_layers,
        self.num_units,
        mode,
        dropout=self.dropout,
        residual_connections=self.residual_connections,
        cell_class=self.cell_class)

    initial_state = self._init_state(
        cell.zero_state(batch_size, tf.float32), initial_state=initial_state)

    return cell, initial_state

  def _build_output_layer(self, vocab_size):
    return tf.layers.Dense(vocab_size, use_bias=True)

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
    _ = memory
    _ = memory_sequence_length

    batch_size = tf.shape(inputs)[0]

    if (sampling_probability is not None
        and (tf.contrib.framework.is_tensor(sampling_probability)
             or sampling_probability > 0.0)):
      if embedding is None:
        raise ValueError("embedding argument must be set when using scheduled sampling")

      tf.summary.scalar("sampling_probability", sampling_probability)
      helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
          inputs,
          sequence_length,
          embedding,
          sampling_probability)
    else:
      helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)

    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state,
        output_layer=output_layer)

    outputs, state, length = tf.contrib.seq2seq.dynamic_decode(decoder)
    return (outputs.rnn_output, state, length)

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

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding,
        start_tokens,
        end_token)

    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state,
        output_layer=output_layer)

    outputs, state, length = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=maximum_iterations)

    predicted_ids = outputs.sample_id
    log_probs = logits_to_cum_log_probs(outputs.rnn_output, length)

    # Make shape consistent with beam search.
    predicted_ids = tf.expand_dims(predicted_ids, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    return (predicted_ids, state, length, log_probs)

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
    batch_size = tf.shape(start_tokens)[0]

    # Replicate batch `beam_width` times.
    if initial_state is not None:
      initial_state = tf.contrib.seq2seq.tile_batch(
          initial_state, multiplier=beam_width)
    if memory is not None:
      memory = tf.contrib.seq2seq.tile_batch(
          memory, multiplier=beam_width)
    if memory_sequence_length is not None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
          memory_sequence_length, multiplier=beam_width)

    cell, initial_state = self._build_cell(
        mode,
        batch_size * beam_width,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
        cell,
        embedding,
        start_tokens,
        end_token,
        initial_state,
        beam_width,
        output_layer=output_layer,
        length_penalty_weight=length_penalty)

    outputs, beam_state, length = tf.contrib.seq2seq.dynamic_decode(
        decoder, maximum_iterations=maximum_iterations)

    predicted_ids = tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])
    log_probs = beam_state.log_probs
    state = beam_state.cell_state

    return (predicted_ids, state, length, log_probs)


class AttentionalRNNDecoder(RNNDecoder):
  """A `RNNDecoder` with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A `onmt.utils.Bridge` to pass the encoder state to the decoder.
      attention_mechanism_class: A class inheriting from
        `tf.contrib.seq2seq.AttentionMechanism`.
      cell_class: The inner cell class.
      dropout: The probability to drop units in each layer output.
      residual_connections: If `True`, each layer input will be added to its output.
    """
    super(AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)
    self.attention_mechanism_class = attention_mechanism_class

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None):
    attention_mechanism = self.attention_mechanism_class(
        self.num_units,
        memory,
        memory_sequence_length=memory_sequence_length)

    cell, initial_cell_state = RNNDecoder._build_cell(
        self,
        mode,
        batch_size,
        initial_state=initial_state)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        initial_cell_state=initial_cell_state)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=1.0 - self.dropout)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)

    return cell, initial_state
