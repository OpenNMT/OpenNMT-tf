"""Define RNN-based decoders."""

import inspect

import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from opennmt.decoders.decoder import Decoder, logits_to_cum_log_probs, build_output_layer
from opennmt.utils.cell import build_cell
from opennmt.layers.reducer import align_in_time


class RNNDecoder(Decoder):
  """A basic RNN decoder."""

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
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
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
                  memory_sequence_length=None,
                  dtype=None,
                  alignment_history=False):
    _ = memory_sequence_length
    _ = alignment_history

    if memory is None and dtype is None:
      raise ValueError("dtype argument is required when memory is not set")

    cell = build_cell(
        self.num_layers,
        self.num_units,
        mode,
        dropout=self.dropout,
        residual_connections=self.residual_connections,
        cell_class=self.cell_class)

    initial_state = self._init_state(
        cell.zero_state(batch_size, dtype or memory.dtype), initial_state=initial_state)

    return cell, initial_state

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
      fused_projection = False
    else:
      helper = tf.contrib.seq2seq.TrainingHelper(inputs, sequence_length)
      fused_projection = True  # With TrainingHelper, project all timesteps at once.

    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=inputs.dtype,
        alignment_history=return_alignment_history)

    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=inputs.dtype)

    decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state,
        output_layer=output_layer if not fused_projection else None)

    outputs, state, length = tf.contrib.seq2seq.dynamic_decode(decoder)

    if fused_projection and output_layer is not None:
      logits = output_layer(outputs.rnn_output)
    else:
      logits = outputs.rnn_output
    # Make sure outputs have the same time_dim as inputs
    inputs_len = tf.shape(inputs)[1]
    logits = align_in_time(logits, inputs_len)

    if return_alignment_history:
      alignment_history = _get_alignment_history(state)
      if alignment_history is not None:
        alignment_history = align_in_time(alignment_history, inputs_len)
      return (logits, state, length, alignment_history)
    return (logits, state, length)

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

    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
        embedding,
        start_tokens,
        end_token)

    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=return_alignment_history)

    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype or memory.dtype)

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
    length = tf.expand_dims(length, 1)
    log_probs = tf.expand_dims(log_probs, 1)

    if return_alignment_history:
      alignment_history = _get_alignment_history(state)
      if alignment_history is not None:
        alignment_history = tf.expand_dims(alignment_history, 1)
      return (predicted_ids, state, length, log_probs, alignment_history)
    return (predicted_ids, state, length, log_probs)

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
    if (return_alignment_history and
        "reorder_tensor_arrays" not in fn_args(tf.contrib.seq2seq.BeamSearchDecoder.__init__)):
      tf.logging.warn("The current version of tf.contrib.seq2seq.BeamSearchDecoder "
                      "does not support returning the alignment history. None will "
                      "be returned instead. Consider upgrading TensorFlow.")
      alignment_history = False
    else:
      alignment_history = return_alignment_history

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
        memory_sequence_length=memory_sequence_length,
        dtype=dtype,
        alignment_history=alignment_history)

    if output_layer is None:
      output_layer = build_output_layer(self.num_units, vocab_size, dtype=dtype or memory.dtype)

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

    if return_alignment_history:
      alignment_history = _get_alignment_history(state)
      if alignment_history is not None:
        alignment_history = tf.reshape(
            alignment_history, [batch_size, beam_width, -1, tf.shape(memory)[1]])
      return (predicted_ids, state, length, log_probs, alignment_history)
    return (predicted_ids, state, length, log_probs)


def _get_alignment_history(cell_state):
  """Returns the alignment history from the cell state."""
  if not hasattr(cell_state, "alignment_history") or cell_state.alignment_history == ():
    return None
  alignment_history = cell_state.alignment_history
  if isinstance(alignment_history, tf.TensorArray):
    alignment_history = alignment_history.stack()
  alignment_history = tf.transpose(alignment_history, perm=[1, 0, 2])
  return alignment_history

def _build_attention_mechanism(attention_mechanism,
                               num_units,
                               memory,
                               memory_sequence_length=None):
  """Builds an attention mechanism from a class or a callable."""
  if inspect.isclass(attention_mechanism):
    kwargs = {}
    if "dtype" in fn_args(attention_mechanism):
      # For TensorFlow 1.5+, dtype should be set in the constructor.
      kwargs["dtype"] = memory.dtype
    return attention_mechanism(
        num_units, memory, memory_sequence_length=memory_sequence_length, **kwargs)
  elif callable(attention_mechanism):
    return attention_mechanism(
        num_units, memory, memory_sequence_length)
  else:
    raise ValueError("Unable to build the attention mechanism")


class AttentionalRNNDecoder(RNNDecoder):
  """A RNN decoder with attention.

  It simple overrides the cell construction to add an attention wrapper.
  """

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               output_is_attention=True,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      attention_mechanism_class: A class inheriting from
        ``tf.contrib.seq2seq.AttentionMechanism`` or a callable that takes
        ``(num_units, memory, memory_sequence_length)`` as arguments and returns
        a ``tf.contrib.seq2seq.AttentionMechanism``.
      output_is_attention: If ``True``, the final decoder output (before logits)
        is the output of the attention layer. In all cases, the output of the
        attention layer is passed to the next step.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
    super(AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)
    self.attention_mechanism_class = attention_mechanism_class
    self.output_is_attention = output_is_attention

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None,
                  alignment_history=False):
    attention_mechanism = _build_attention_mechanism(
        self.attention_mechanism_class,
        self.num_units,
        memory,
        memory_sequence_length=memory_sequence_length)

    cell, initial_cell_state = RNNDecoder._build_cell(
        self,
        mode,
        batch_size,
        initial_state=initial_state,
        dtype=memory.dtype)

    cell = tf.contrib.seq2seq.AttentionWrapper(
        cell,
        attention_mechanism,
        attention_layer_size=self.num_units,
        alignment_history=alignment_history,
        output_attention=self.output_is_attention,
        initial_cell_state=initial_cell_state)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(
          cell, output_keep_prob=1.0 - self.dropout)

    initial_state = cell.zero_state(batch_size, memory.dtype)

    return cell, initial_state


class MultiAttentionalRNNDecoder(RNNDecoder):
  """A RNN decoder with multi-attention.

  This decoder can attend the encoder outputs after multiple RNN layers using
  one or multiple attention mechanisms. Additionally, the cell state of this
  decoder is not initialized from the encoder state (i.e. a
  :class:`opennmt.layers.bridge.ZeroBridge` is imposed).
  """

  def __init__(self,
               num_layers,
               num_units,
               attention_layers=None,
               attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      attention_layers: A list of integers, the layers after which to add
        attention. If ``None``, attention will only be added after the last
        layer.
      attention_mechanism_class: A class or list of classes inheriting from
        ``tf.contrib.seq2seq.AttentionMechanism``. Alternatively, the class can
        be replaced by a callable that takes
        ``(num_units, memory, memory_sequence_length)`` as arguments and returns
        a ``tf.contrib.seq2seq.AttentionMechanism``.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
    super(MultiAttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)

    attention_layers = attention_layers or [-1]
    attention_layers = [l % num_layers for l in attention_layers]

    if not isinstance(attention_mechanism_class, list):
      attention_mechanism_class = [attention_mechanism_class for _ in attention_layers]

    self.attention_mechanism_class = attention_mechanism_class
    self.attention_layers = attention_layers

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None,
                  alignment_history=False):
    attention_mechanisms = [
        _build_attention_mechanism(
            attention_mechanism,
            self.num_units,
            memory,
            memory_sequence_length=memory_sequence_length)
        for attention_mechanism in self.attention_mechanism_class]

    cell = build_cell(
        self.num_layers,
        self.num_units,
        mode,
        dropout=self.dropout,
        residual_connections=self.residual_connections,
        cell_class=self.cell_class,
        attention_layers=self.attention_layers,
        attention_mechanisms=attention_mechanisms)

    initial_state = cell.zero_state(batch_size, memory.dtype)

    return cell, initial_state
