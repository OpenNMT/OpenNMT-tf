"""Define RNN-based encoders."""

import tensorflow as tf

from opennmt.v2.encoders.encoder import Encoder
from opennmt.v2.layers import rnn
from opennmt.layers import reducer as reducers


class UnidirectionalRNNEncoder(Encoder):
  """A simple RNN encoder."""

  def __init__(self,
               num_layers,
               num_units,
               cell_class=tf.keras.layers.LSTMCell,
               dropout=0.3,
               residual_connections=False,
               **kwargs):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.
      kwargs: Additional layer arguments.
    """
    super(UnidirectionalRNNEncoder, self).__init__(**kwargs)
    cell = rnn.make_rnn_cell(
        num_layers,
        num_units,
        dropout=dropout,
        residual_connections=residual_connections,
        cell_class=cell_class)
    self.rnn = rnn.RNN(cell)

  def encode(self, inputs, sequence_length=None, training=None):
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    outputs, states = self.rnn(inputs, mask=mask, training=training)
    return outputs, states, sequence_length


class BidirectionalRNNEncoder(Encoder):
  """An encoder that encodes an input sequence in both directions."""

  def __init__(self,
               num_layers,
               num_units,
               reducer=reducers.SumReducer(),
               cell_class=tf.keras.layers.LSTMCell,
               dropout=0.3,
               residual_connections=False,
               **kwargs):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      reducer: A :class:`opennmt.layers.reducer.Reducer` instance to merge
        bidirectional state and outputs.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its
        output.

    Raises:
      ValueError: when using :class:`opennmt.layers.reducer.ConcatReducer` and
        :obj:`num_units` is not divisible by 2.
    """
    if isinstance(reducer, reducers.ConcatReducer):
      if num_units % 2 != 0:
        raise ValueError("num_units must be divisible by 2 to use the ConcatReducer.")
      num_units /= 2
    super(BidirectionalRNNEncoder, self).__init__(**kwargs)
    cell = rnn.make_rnn_cell(
        num_layers,
        num_units,
        dropout=dropout,
        residual_connections=residual_connections,
        cell_class=cell_class)
    self.rnn = rnn.RNN(cell, bidirectional=True, reducer=reducer)

  def encode(self, inputs, sequence_length=None, training=None):
    mask = self.build_mask(inputs, sequence_length=sequence_length)
    outputs, states = self.rnn(inputs, mask=mask, training=training)
    return outputs, states, sequence_length
