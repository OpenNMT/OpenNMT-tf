# pylint: disable=W0223

"""Define RNN-based decoders."""

import inspect

import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from opennmt.decoders import decoder
from opennmt.utils.cell import build_cell
from opennmt.layers.reducer import align_in_time
from opennmt.layers.transformer import build_sequence_mask, multi_head_attention


class RNNDecoder(decoder.Decoder):
  """A basic RNN decoder."""

  def __init__(self,
               num_layers,
               num_units,
               bridge=None,
               cell_class=None,
               dropout=0.3,
               residual_connections=False):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bridge: A :class:`opennmt.layers.bridge.Bridge` to pass the encoder state
        to the decoder.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
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

  @property
  def output_size(self):
    """Returns the decoder output size."""
    return self.num_units

  def _init_state(self, zero_state, initial_state=None):
    if initial_state is None:
      return zero_state
    elif self.bridge is None:
      raise ValueError("A bridge must be configured when passing encoder state")
    else:
      return self.bridge(initial_state, zero_state)

  def _get_attention(self, state, step=None):  # pylint: disable=unused-argument
    return None

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None):
    _ = memory_sequence_length

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
        dtype=inputs.dtype)

    if output_layer is None:
      output_layer = decoder.build_output_layer(
          self.output_size, vocab_size, dtype=inputs.dtype)

    basic_decoder = tf.contrib.seq2seq.BasicDecoder(
        cell,
        helper,
        initial_state,
        output_layer=output_layer if not fused_projection else None)

    outputs, state, length = tf.contrib.seq2seq.dynamic_decode(basic_decoder)

    if fused_projection and output_layer is not None:
      logits = output_layer(outputs.rnn_output)
    else:
      logits = outputs.rnn_output
    # Make sure outputs have the same time_dim as inputs
    inputs_len = tf.shape(inputs)[1]
    logits = align_in_time(logits, inputs_len)

    if return_alignment_history:
      alignment_history = self._get_attention(state)
      if alignment_history is not None:
        alignment_history = align_in_time(alignment_history, inputs_len)
      return (logits, state, length, alignment_history)
    return (logits, state, length)

  def step_fn(self,
              mode,
              batch_size,
              initial_state=None,
              memory=None,
              memory_sequence_length=None,
              dtype=tf.float32):
    cell, initial_state = self._build_cell(
        mode,
        batch_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        dtype=dtype)

    def _fn(step, inputs, state, mode):
      _ = mode
      # This scope is defined by tf.contrib.seq2seq.dynamic_decode during the
      # training.
      with tf.variable_scope("decoder"):
        outputs, state = cell(inputs, state)
        if self.support_alignment_history:
          return outputs, state, self._get_attention(state, step=step)
        return outputs, state

    return _fn, initial_state


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
               attention_mechanism_class=None,
               output_is_attention=True,
               cell_class=None,
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
        a ``tf.contrib.seq2seq.AttentionMechanism``. Defaults to
        ``tf.contrib.seq2seq.LuongAttention``.
      output_is_attention: If ``True``, the final decoder output (before logits)
        is the output of the attention layer. In all cases, the output of the
        attention layer is passed to the next step.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
    if attention_mechanism_class is None:
      attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
    super(AttentionalRNNDecoder, self).__init__(
        num_layers,
        num_units,
        bridge=bridge,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)
    self.attention_mechanism_class = attention_mechanism_class
    self.output_is_attention = output_is_attention

  @property
  def support_alignment_history(self):
    return True

  def _get_attention(self, state, step=None):
    alignment_history = state.alignment_history
    if step is not None:
      return alignment_history.read(step)
    return tf.transpose(alignment_history.stack(), perm=[1, 0, 2])

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None):
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
        alignment_history=True,
        output_attention=self.output_is_attention,
        initial_cell_state=initial_cell_state)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout > 0.0:
      cell = tf.nn.rnn_cell.DropoutWrapper(
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
               attention_mechanism_class=None,
               cell_class=None,
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
        a ``tf.contrib.seq2seq.AttentionMechanism``. Defaults to
        ``tf.contrib.seq2seq.LuongAttention``.
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
    if attention_mechanism_class is None:
      attention_mechanism_class = tf.contrib.seq2seq.LuongAttention
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
                  dtype=None):
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


class RNMTPlusDecoder(RNNDecoder):
  """The RNMT+ decoder described in https://arxiv.org/abs/1804.09849."""

  def __init__(self,
               num_layers,
               num_units,
               num_heads,
               cell_class=None,
               dropout=0.3):
    """Initializes the decoder parameters.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      num_heads: The number of attention heads.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a layer normalized LSTM cell.
      dropout: The probability to drop units from the decoder input and in each
        layer output.
    """
    if cell_class is None:
      cell_class = tf.contrib.rnn.LayerNormBasicLSTMCell
    super(RNMTPlusDecoder, self).__init__(
        num_layers,
        num_units,
        cell_class=cell_class,
        dropout=dropout)
    self.num_heads = num_heads

  @property
  def output_size(self):
    """Returns the decoder output size."""
    return self.num_units * 2

  def _build_cell(self,
                  mode,
                  batch_size,
                  initial_state=None,
                  memory=None,
                  memory_sequence_length=None,
                  dtype=None):
    cell = _RNMTPlusDecoderCell(
        mode,
        self.num_layers,
        self.num_units,
        self.num_heads,
        memory,
        memory_sequence_length,
        cell_class=self.cell_class,
        dropout=self.dropout)
    return cell, cell.zero_state(batch_size, dtype)

class _RNMTPlusDecoderCell(tf.nn.rnn_cell.RNNCell):

  def __init__(self,
               mode,
               num_layers,
               num_units,
               num_heads,
               memory,
               memory_sequence_length,
               cell_class=None,
               dropout=0.3):
    super(_RNMTPlusDecoderCell, self).__init__()
    self._mode = mode
    self._num_units = num_units
    self._num_heads = num_heads
    self._dropout = dropout
    self._cells = [cell_class(num_units) for _ in range(num_layers)]
    self._memory = memory
    self._memory_mask = build_sequence_mask(
        memory_sequence_length,
        num_heads=self._num_heads,
        maximum_length=tf.shape(memory)[1])

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  @property
  def output_size(self):
    return self._num_units * 2

  def zero_state(self, batch_size, dtype):
    with tf.name_scope("RNMTPlusDecoderCellZeroState", values=[batch_size]):
      return tuple(cell.zero_state(batch_size, dtype) for cell in self._cells)

  def __call__(self, inputs, state, scope=None):
    inputs = tf.layers.dropout(
        inputs, rate=self._dropout, training=self._mode == tf.estimator.ModeKeys.TRAIN)

    new_states = []
    with tf.variable_scope("rnn_0"):
      last_outputs, state_0 = self._cells[0](inputs, state[0])
      new_states.append(state_0)

    with tf.variable_scope("multi_head_attention"):
      context = multi_head_attention(
          self._num_heads,
          tf.expand_dims(last_outputs, 1),
          self._memory,
          self._mode,
          mask=self._memory_mask,
          dropout=self._dropout)
      context = tf.squeeze(context, axis=1)

    for i in range(1, len(self._cells)):
      inputs = tf.concat([last_outputs, context], axis=-1)
      with tf.variable_scope("rnn_%d" % i):
        outputs, state_i = self._cells[i](inputs, state[i])
        new_states.append(state_i)
        outputs = tf.layers.dropout(
            outputs, rate=self._dropout, training=self._mode == tf.estimator.ModeKeys.TRAIN)
        if i >= 2:
          outputs += last_outputs
        last_outputs = outputs

    final = tf.concat([last_outputs, context], -1)
    return final, tuple(new_states)
