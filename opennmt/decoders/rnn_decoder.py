"""Define RNN-based decoders."""

import tensorflow as tf

from tensorflow.python.layers.core import Dense

from opennmt.decoders.decoder import Decoder
from opennmt.utils.cell import build_cell


class RNNDecoder(Decoder):
  """A basic RNN-based decoder."""

  def __init__(self,
               num_layers,
               num_units,
               bridge,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A `onmt.utils.Bridge` to pass the encoder states to the decoder.
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

  def _build_cell(self,
                  mode,
                  batch_size,
                  encoder_states=None,
                  memory=None,
                  memory_sequence_length=None):
    cell = build_cell(
      self.num_layers,
      self.num_units,
      mode,
      dropout=self.dropout,
      residual_connections=self.residual_connections,
      cell_class=self.cell_class)

    initial_state = cell.zero_state(batch_size, tf.float32)

    if not encoder_states is None:
      initial_state = self.bridge(encoder_states, initial_state)

    return cell, initial_state

  def _build_output_layer(self, vocab_size):
    return Dense(vocab_size, use_bias=True)

  def decode(self,
             inputs,
             sequence_length,
             vocab_size,
             encoder_states=None,
             scheduled_sampling_probability=0.0,
             embeddings=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    batch_size = tf.shape(inputs)[0]

    if scheduled_sampling_probability > 0:
      if embeddings is None:
        raise ValueError("embeddings argument must be set when using scheduled sampling")

      helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
        inputs,
        sequence_length,
        embeddings,
        scheduled_sampling_probability)
    else:
      helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)

    cell, initial_state = self._build_cell(
      mode,
      batch_size,
      encoder_states=encoder_states,
      memory=memory,
      memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BasicDecoder(
      cell,
      helper,
      initial_state,
      output_layer=output_layer)

    outputs, states, length = tf.contrib.seq2seq.dynamic_decode(decoder)
    return (outputs.rnn_output, states, length)

  def dynamic_decode(self,
                     embeddings,
                     start_tokens,
                     end_token,
                     vocab_size,
                     encoder_states=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None):
    batch_size = tf.shape(start_tokens)[0]

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
      embeddings,
      start_tokens,
      end_token)

    cell, initial_state = self._build_cell(
      mode,
      batch_size,
      encoder_states=encoder_states,
      memory=memory,
      memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BasicDecoder(
      cell,
      helper,
      initial_state,
      output_layer=output_layer)

    outputs, states, length = tf.contrib.seq2seq.dynamic_decode(
      decoder, maximum_iterations=maximum_iterations)

    predicted_ids = outputs.sample_id
    predicted_ids = tf.expand_dims(predicted_ids, 1)

    return (predicted_ids, states, length)

  def dynamic_decode_and_search(self,
                                embeddings,
                                start_tokens,
                                end_token,
                                vocab_size,
                                encoder_states=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None):
    batch_size = tf.shape(start_tokens)[0]

    # Replicate batch `beam_width` times.
    if not encoder_states is None:
      encoder_states = tf.contrib.seq2seq.tile_batch(
        encoder_states, multiplier=beam_width)
    if not memory is None:
      memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    if not memory_sequence_length is None:
      memory_sequence_length = tf.contrib.seq2seq.tile_batch(
        memory_sequence_length, multiplier=beam_width)

    cell, initial_state = self._build_cell(
      mode,
      batch_size * beam_width,
      encoder_states=encoder_states,
      memory=memory,
      memory_sequence_length=memory_sequence_length)

    output_layer = self._build_output_layer(vocab_size)

    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
      cell,
      embeddings,
      start_tokens,
      end_token,
      initial_state,
      beam_width,
      output_layer=output_layer,
      length_penalty_weight=length_penalty)

    outputs, states, length = tf.contrib.seq2seq.dynamic_decode(
      decoder, maximum_iterations=maximum_iterations)
    predicted_ids = tf.transpose(outputs.predicted_ids, perm=[0, 2, 1])

    return (predicted_ids, states, length)


class AttentionalRNNDecoder(RNNDecoder):
  """A `RNNDecoder` with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A `onmt.utils.Bridge` to pass the encoder states to the decoder.
      attention_mechanism_class: A class inheriting from
        `tf.contrib.seq2seq.AttentionMechanism`.
      cell_class: The inner cell class.
      dropout: The probability to drop units in each layer output.
      residual_connections: If `True`, each layer input will be added to its output.
    """
    super(AttentionalRNNDecoder, self).__init__(num_layers,
                                                num_units,
                                                bridge,
                                                cell_class=cell_class,
                                                dropout=dropout,
                                                residual_connections=residual_connections)
    self.attention_mechanism_class = attention_mechanism_class

  def _build_cell(self,
                  mode,
                  batch_size,
                  encoder_states=None,
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
      encoder_states=encoder_states)

    if not encoder_states is None:
      initial_cell_state = self.bridge(encoder_states, initial_cell_state)

    cell = tf.contrib.seq2seq.AttentionWrapper(
      cell,
      attention_mechanism,
      initial_cell_state=initial_cell_state)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=1.0 - self.dropout)

    initial_state = cell.zero_state(batch_size, dtype=tf.float32)

    return cell, initial_state
