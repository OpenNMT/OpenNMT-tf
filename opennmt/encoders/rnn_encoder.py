"""Define RNN-based encoders."""

import abc
import six

import tensorflow as tf

from opennmt.utils.cell import build_cell
from opennmt.encoders.encoder import Encoder
from opennmt.utils.reducer import SumReducer, ConcatReducer


@six.add_metaclass(abc.ABCMeta)
class RNNEncoder(Encoder):
  """Base class for RNN encoders."""

  def __init__(self,
               num_layers,
               num_units,
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.0,
               residual_connections=False):
    """Common constructor to save parameters."""
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
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      cell_class: The inner cell class.
      dropout: The probability to drop units in each layer output.
      residual_connections: If `True`, each layer input will be added to its output.
    """
    super(UnidirectionalRNNEncoder, self).__init__(
      num_layers,
      num_units,
      cell_class=cell_class,
      dropout=dropout,
      residual_connections=residual_connections)

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    cell = self._build_cell(mode)

    encoder_outputs, encoder_states = tf.nn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=sequence_length,
      dtype=tf.float32)

    return (encoder_outputs, encoder_states, sequence_length)


class BidirectionalRNNEncoder(RNNEncoder):
  """An encoder that encodes an input sequence in both directions."""

  def __init__(self,
               num_layers,
               num_units,
               reducer=SumReducer(),
               cell_class=tf.contrib.rnn.LSTMCell,
               dropout=0.3,
               residual_connections=False):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      reducer: A `StatesReducer` instance to merge bidirectional states and outputs.
      cell_class: The inner cell class.
      dropout: The probability to drop units in each layer output.
      residual_connections: If `True`, each layer input will be added to its output.
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
    with tf.variable_scope("fw"):
      cell_fw = self._build_cell(mode)
    with tf.variable_scope("bw"):
      cell_bw = self._build_cell(mode)

    encoder_outputs_tup, encoder_states_tup = tf.nn.bidirectional_dynamic_rnn(
      cell_fw,
      cell_bw,
      inputs,
      sequence_length=sequence_length,
      dtype=tf.float32)

    # Merge bidirectional outputs and states.
    encoder_outputs = self.reducer.reduce(encoder_outputs_tup[0], encoder_outputs_tup[1])
    encoder_states = self.reducer.reduce(encoder_states_tup[0], encoder_states_tup[1])

    return (encoder_outputs, encoder_states, sequence_length)


class GoogleRNNEncoder(Encoder):
  """The RNN encoder used in GNMT as described in https://arxiv.org/abs/1609.08144."""

  def __init__(self,
               num_layers,
               num_units,
               dropout=0.3):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      dropout: The probability to drop units in each layer output.
    """
    if num_layers < 2:
      raise ValueError("GoogleRNNEncoder requires at least 2 layers")

    self.bidirectional = BidirectionalRNNEncoder(
      1,
      num_units,
      reducer=ConcatReducer(),
      cell_class=tf.contrib.rnn.LSTMCell,
      dropout=dropout)
    self.unidirectional = UnidirectionalRNNEncoder(
      num_layers - 1,
      num_units,
      cell_class=tf.contrib.rnn.LSTMCell,
      dropout=dropout,
      residual_connections=True)

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_outputs, bi_states, sequence_length = self.bidirectional.encode(
      inputs,
      sequence_length=sequence_length,
      mode=mode)
    encoder_outputs, uni_states, sequence_length = self.unidirectional.encode(
      encoder_outputs,
      sequence_length=sequence_length,
      mode=mode)

    encoder_states = bi_states + uni_states

    return (encoder_outputs, encoder_states, sequence_length)
