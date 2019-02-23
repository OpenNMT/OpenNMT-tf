"""Define RNN-based encoders."""

import abc
import six

import tensorflow as tf

from opennmt.utils.cell import build_cell
from opennmt.encoders.encoder import Encoder, SequentialEncoder
from opennmt.layers.reducer import SumReducer, ConcatReducer, JoinReducer, pad_in_time
from opennmt.layers import rnn


@six.add_metaclass(abc.ABCMeta)
class RNNEncoder(Encoder):
  """Base class for RNN encoders."""

  def __init__(self,
               num_layers,
               num_units,
               cell_class=None,
               dropout=0.0,
               residual_connections=False):
    """Common constructor to save parameters."""
    super(RNNEncoder, self).__init__()
    self.num_layers = num_layers
    self.num_units = num_units
    self.cell_class = cell_class
    self.dropout = dropout
    self.residual_connections = residual_connections

  def _build_cell(self, mode):
    return build_cell(
        self.num_layers,
        self.num_units,
        mode,
        dropout=self.dropout,
        residual_connections=self.residual_connections,
        cell_class=self.cell_class)

  @abc.abstractmethod
  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    raise NotImplementedError()


class UnidirectionalRNNEncoder(RNNEncoder):
  """A simple RNN encoder."""

  def __init__(self,
               num_layers,
               num_units,
               cell_class=None,
               dropout=0.3,
               residual_connections=False):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
    """
    super(UnidirectionalRNNEncoder, self).__init__(
        num_layers,
        num_units,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    cell = self._build_cell(mode)

    encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
        cell,
        inputs,
        sequence_length=sequence_length,
        dtype=inputs.dtype)

    return (encoder_outputs, encoder_state, sequence_length)


class BidirectionalRNNEncoder(RNNEncoder):
  """An encoder that encodes an input sequence in both directions."""

  def __init__(self,
               num_layers,
               num_units,
               reducer=SumReducer(),
               cell_class=None,
               dropout=0.3,
               residual_connections=False):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      reducer: A :class:`opennmt.layers.reducer.Reducer` instance to merge
        bidirectional state and outputs.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.

    Raises:
      ValueError: when using :class:`opennmt.layers.reducer.ConcatReducer` and
        :obj:`num_units` is not divisible by 2.
    """
    if isinstance(reducer, ConcatReducer):
      if num_units % 2 != 0:
        raise ValueError("num_units must be divisible by 2 to use the ConcatReducer.")
      num_units /= 2

    self.reducer = reducer

    super(BidirectionalRNNEncoder, self).__init__(
        num_layers,
        num_units,
        cell_class=cell_class,
        dropout=dropout,
        residual_connections=residual_connections)

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    cell_fw = self._build_cell(mode)
    cell_bw = self._build_cell(mode)

    encoder_outputs_tup, encoder_state_tup = tf.nn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs,
        sequence_length=sequence_length,
        dtype=inputs.dtype)

    # Merge bidirectional outputs and state.
    encoder_outputs = self.reducer.zip_and_reduce(encoder_outputs_tup[0], encoder_outputs_tup[1])
    encoder_state = self.reducer.zip_and_reduce(encoder_state_tup[0], encoder_state_tup[1])

    return (encoder_outputs, encoder_state, sequence_length)


class RNMTPlusEncoder(Encoder):
  """The RNMT+ encoder described in https://arxiv.org/abs/1804.09849."""

  def __init__(self,
               num_layers=6,
               num_units=1024,
               cell_class=None,
               dropout=0.3):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each RNN layer and the final output.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a layer normalized LSTM cell.
        For efficiency, consider using the standard ``tf.nn.rnn_cell.LSTMCell``
        instead.
      dropout: The probability to drop units in each layer output.
    """
    super(RNMTPlusEncoder, self).__init__()
    if cell_class is None:
      cell_class = tf.contrib.rnn.LayerNormBasicLSTMCell
    self._num_units = num_units
    self._dropout = dropout
    self._layers = [
        BidirectionalRNNEncoder(
            num_layers=1,
            num_units=num_units * 2,
            reducer=ConcatReducer(),
            cell_class=cell_class,
            dropout=0.0)
        for _ in range(num_layers)]

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    inputs = tf.layers.dropout(
        inputs,
        rate=self._dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    states = []
    for i, layer in enumerate(self._layers):
      with tf.variable_scope("layer_%d" % i):
        outputs, state, sequence_length = layer.encode(
            inputs, sequence_length=sequence_length, mode=mode)
        outputs = tf.layers.dropout(
            outputs,
            rate=self._dropout,
            training=mode == tf.estimator.ModeKeys.TRAIN)
        inputs = outputs + inputs if i >= 2 else outputs
        states.append(state)

    with tf.variable_scope("projection"):
      projected = tf.layers.dense(inputs, self._num_units)
    state = JoinReducer()(states)
    return (projected, state, sequence_length)


class GoogleRNNEncoder(Encoder):
  """The RNN encoder used in GNMT as described in
  https://arxiv.org/abs/1609.08144.
  """

  def __init__(self,
               num_layers,
               num_units,
               dropout=0.3):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      dropout: The probability to drop units in each layer output.

    Raises:
      ValueError: if :obj:`num_layers` < 2.
    """
    super(GoogleRNNEncoder, self).__init__()
    if num_layers < 2:
      raise ValueError("GoogleRNNEncoder requires at least 2 layers")

    self.bidirectional = BidirectionalRNNEncoder(
        1,
        num_units,
        reducer=ConcatReducer(),
        cell_class=tf.nn.rnn_cell.LSTMCell,
        dropout=dropout)
    self.unidirectional = UnidirectionalRNNEncoder(
        num_layers - 1,
        num_units,
        cell_class=tf.nn.rnn_cell.LSTMCell,
        dropout=dropout,
        residual_connections=True)

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_outputs, bidirectional_state, sequence_length = self.bidirectional.encode(
        inputs,
        sequence_length=sequence_length,
        mode=mode)
    encoder_outputs, unidirectional_state, sequence_length = self.unidirectional.encode(
        encoder_outputs,
        sequence_length=sequence_length,
        mode=mode)

    encoder_state = JoinReducer()([bidirectional_state, unidirectional_state])

    return (encoder_outputs, encoder_state, sequence_length)


class PyramidalRNNEncoder(Encoder):
  """An encoder that reduces the time dimension after each bidirectional layer."""

  def __init__(self,
               num_layers,
               num_units,
               reduction_factor=2,
               cell_class=None,
               dropout=0.3):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      reduction_factor: The time reduction factor.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      dropout: The probability to drop units in each layer output.
    """
    super(PyramidalRNNEncoder, self).__init__()
    self.reduction_factor = reduction_factor
    self.state_reducer = JoinReducer()
    self.layers = []

    for _ in range(num_layers):
      self.layers.append(BidirectionalRNNEncoder(
          1,
          num_units,
          reducer=ConcatReducer(),
          cell_class=cell_class,
          dropout=dropout))

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_state = []

    for layer_index, layer in enumerate(self.layers):
      input_depth = inputs.get_shape().as_list()[-1]

      if layer_index == 0:
        # For the first input, make the number of timesteps a multiple of the
        # total reduction factor.
        total_reduction_factor = pow(self.reduction_factor, len(self.layers) - 1)

        current_length = tf.shape(inputs)[1]
        factor = tf.cast(current_length, tf.float32) / total_reduction_factor
        new_length = tf.cast(tf.ceil(factor), tf.int32) * total_reduction_factor
        inputs = pad_in_time(inputs, new_length - current_length)

        # Lengths should not be smaller than the total reduction factor.
        sequence_length = tf.maximum(sequence_length, total_reduction_factor)
      else:
        # In other cases, reduce the time dimension.
        inputs = tf.reshape(
            inputs,
            [tf.shape(inputs)[0], -1, input_depth * self.reduction_factor])
        if sequence_length is not None:
          sequence_length //= self.reduction_factor

      with tf.variable_scope("layer_{}".format(layer_index)):
        outputs, state, sequence_length = layer.encode(
            inputs,
            sequence_length=sequence_length,
            mode=mode)

      encoder_state.append(state)
      inputs = outputs

    return (
        outputs,
        self.state_reducer(encoder_state),
        sequence_length)


class RNNEncoderV2(Encoder):
  """A RNN sequence encoder.

  Note:
    TensorFlow 2.0 version.
  """

  def __init__(self,
               num_layers,
               num_units,
               bidirectional=False,
               residual_connections=False,
               dropout=0.3,
               reducer=ConcatReducer(),
               cell_class=None,
               **kwargs):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      bidirectional: Use a bidirectional RNN.
      residual_connections: If ``True``, each layer input will be added to its
        output.
      reducer: A :class:`opennmt.layers.reducer.Reducer` instance to merge
        bidirectional state and outputs.
      dropout: The probability to drop units in each layer output.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      **kwargs: Additional layer arguments.
    """
    super(RNNEncoderV2, self).__init__(**kwargs)
    cell = rnn.make_rnn_cell(
        num_layers,
        num_units,
        dropout=dropout,
        residual_connections=residual_connections,
        cell_class=cell_class)
    self.rnn = rnn.RNN(cell, bidirectional=bidirectional, reducer=reducer)

  def call(self, inputs, sequence_length=None, training=None):
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    outputs, states = self.rnn(inputs, mask=mask, training=training)
    return outputs, states, sequence_length


class GNMTEncoder(SequentialEncoder):
  """The RNN encoder used in GNMT as described in
  https://arxiv.org/abs/1609.08144.

  Note:
    TensorFlow 2.0 version.
  """

  def __init__(self, num_layers, num_units, dropout=0.3):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      dropout: The probability to drop units in each layer output.

    Raises:
      ValueError: if :obj:`num_layers` < 2.
    """
    if num_layers < 2:
      raise ValueError("GoogleRNNEncoder requires at least 2 layers")
    bidirectional = RNNEncoderV2(
        1,
        num_units // 2,
        bidirectional=True,
        dropout=dropout)
    unidirectional = RNNEncoderV2(
        num_layers - 1,
        num_units,
        dropout=dropout,
        residual_connections=True)
    super(GNMTEncoder, self).__init__([bidirectional, unidirectional])
