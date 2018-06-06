"""RNN cells helpers."""

import collections

import tensorflow as tf


def build_cell(num_layers,
               num_units,
               mode,
               dropout=0.0,
               residual_connections=False,
               cell_class=tf.contrib.rnn.LSTMCell,
               attention_layers=None,
               attention_mechanisms=None):
  """Convenience function to build a multi-layer RNN cell.

  Args:
    num_layers: The number of layers.
    num_units: The number of units in each layer.
    mode: A ``tf.estimator.ModeKeys`` mode.
    dropout: The probability to drop units in each layer output.
    residual_connections: If ``True``, each layer input will be added to its output.
    cell_class: The inner cell class or a callable taking :obj:`num_units` as
      argument and returning a cell.
    attention_layers: A list of integers, the layers after which to add attention.
    attention_mechanisms: A list of ``tf.contrib.seq2seq.AttentionMechanism``
      with the same length as :obj:`attention_layers`.

  Returns:
    A ``tf.contrib.rnn.RNNCell``.

  Raises:
    ValueError: if :obj:`attention_layers` and :obj:`attention_mechanisms` do
      not have the same length.
  """
  cells = []

  attention_mechanisms = attention_mechanisms or []
  attention_layers = attention_layers or []

  if len(attention_mechanisms) != len(attention_layers):
    raise ValueError("There must be the same number of attention mechanisms "
                     "as the number of attention layers")

  for l in range(num_layers):
    cell = cell_class(num_units)
    if l in attention_layers:
      cell = tf.contrib.seq2seq.AttentionWrapper(
          cell,
          attention_mechanisms[attention_layers.index(l)],
          attention_layer_size=num_units)
    if mode == tf.estimator.ModeKeys.TRAIN and dropout > 0.0:
      cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=1.0 - dropout)
    if residual_connections and l > 0:
      cell = tf.contrib.rnn.ResidualWrapper(cell)
    cells.append(cell)

  if len(cells) == 1:
    return cells[0]
  else:
    return tf.contrib.rnn.MultiRNNCell(cells)

def last_encoding_from_state(state):
  """Returns the last encoding vector from the state.

  For example, this is the last hidden states of the last LSTM layer for a
  LSTM-based encoder.

  Args:
    state: The encoder state.

  Returns:
    The last encoding vector.
  """
  if isinstance(state, collections.Sequence):
    state = state[-1]
  if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
    return state.h
  return state
