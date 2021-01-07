"""RNN functions and classes for TensorFlow 2.0."""

import tensorflow as tf
import numpy as np

from opennmt.layers import common
from opennmt.layers import reducer as reducer_lib


class RNNCellWrapper(common.LayerWrapper):
    """A wrapper for RNN cells."""

    def __init__(
        self,
        cell,
        input_dropout=0,
        output_dropout=0,
        residual_connection=False,
        **kwargs
    ):
        """Initializes the wrapper.

        Args:
          cell: The cell to wrap.
          input_dropout: The probability to drop units in the cell input.
          output_dropout: The probability to drop units in the cell output.
          residual_connection: Add the inputs to cell outputs (if their shape are
            compatible).
          kwargs: Additional layer arguments.
        """
        super().__init__(
            cell,
            input_dropout=input_dropout,
            output_dropout=output_dropout,
            residual_connection=residual_connection,
            **kwargs,
        )
        self.cell = cell

    @property
    def state_size(self):
        """The cell state size."""
        return self.cell.state_size

    @property
    def output_size(self):
        """The cell output size."""
        return self.cell.output_size

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """Returns the initial cell state."""
        return self.cell.get_initial_state(
            inputs=inputs, batch_size=batch_size, dtype=dtype
        )


def make_rnn_cell(
    num_layers,
    num_units,
    dropout=0,
    residual_connections=False,
    cell_class=None,
    **kwargs
):
    """Convenience function to build a multi-layer RNN cell.

    Args:
      num_layers: The number of layers.
      num_units: The number of units in each layer.
      dropout: The probability to drop units in each layer output.
      residual_connections: If ``True``, each layer input will be added to its output.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      kwargs: Additional arguments passed to the cell constructor.

    Returns:
      A ``tf.keras.layers.StackedRNNCells`` instance.

    See Also:
      :class:`opennmt.layers.RNNCellWrapper`
    """
    if cell_class is None:
        cell_class = tf.keras.layers.LSTMCell
    cells = []
    for _ in range(num_layers):
        cell = cell_class(num_units, **kwargs)
        if dropout > 0 or residual_connections:
            cell = RNNCellWrapper(
                cell, output_dropout=dropout, residual_connection=residual_connections
            )
        cells.append(cell)
    return tf.keras.layers.StackedRNNCells(cells)


class _RNNWrapper(tf.keras.layers.Layer):
    """Extend a RNN layer to possibly make it bidirectional and format its outputs."""

    def __init__(
        self, rnn, bidirectional=False, reducer=reducer_lib.ConcatReducer(), **kwargs
    ):
        """Initializes the layer.

        Args:
          rnn: The RNN layer to extend, built with ``return_sequences`` and
            ``return_state`` enabled.
          bidirectional: Make this layer bidirectional.
          reducer: A :class:`opennmt.layers.Reducer` instance to merge
            bidirectional states and outputs.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.rnn = rnn
        self.reducer = reducer
        self.bidirectional = bidirectional
        if bidirectional:
            self.rnn = tf.keras.layers.Bidirectional(self.rnn, merge_mode=None)

    def call(self, *args, **kwargs):
        """Forwards the arguments to the RNN layer.

        Args:
          *args: Positional arguments of the RNN layer.
          **kwargs: Keyword arguments of the RNN layer.

        Returns:
          A tuple with the output sequences and the states.
        """
        outputs = self.rnn(*args, **kwargs)
        if self.bidirectional:
            sequences = outputs[0:2]
            states = outputs[2:]
            fwd_states = states[: len(states) // 2]
            bwd_states = states[len(states) // 2 :]
            if self.reducer is not None:
                sequences = self.reducer(sequences)
                states = tuple(self.reducer.zip_and_reduce(fwd_states, bwd_states))
            else:
                sequences = tuple(sequences)
                states = (fwd_states, bwd_states)
        else:
            sequences = outputs[0]
            states = tuple(outputs[1:])
        return sequences, states


class RNN(_RNNWrapper):
    """A simple RNN layer."""

    def __init__(
        self, cell, bidirectional=False, reducer=reducer_lib.ConcatReducer(), **kwargs
    ):
        """Initializes the layer.

        Args:
          cell: The RNN cell to use.
          bidirectional: Make this layer bidirectional.
          reducer: A :class:`opennmt.layers.Reducer` instance to merge
            bidirectional states and outputs.
          **kwargs: Additional layer arguments.

        See Also:
          :func:`opennmt.layers.make_rnn_cell`
        """
        rnn = tf.keras.layers.RNN(cell, return_sequences=True, return_state=True)
        super().__init__(rnn, bidirectional=bidirectional, reducer=reducer, **kwargs)

    def map_v1_weights(self, weights):
        m = []
        if self.bidirectional:
            weights = weights["bidirectional_rnn"]
            m += map_v1_weights_to_cell(self.rnn.forward_layer.cell, weights["fw"])
            m += map_v1_weights_to_cell(self.rnn.backward_layer.cell, weights["bw"])
        else:
            weights = weights["rnn"]
            m += map_v1_weights_to_cell(self.rnn.cell, weights)
        return m


class LSTM(tf.keras.layers.Layer):
    """A multi-layer LSTM.

    This differs from using :class:`opennmt.layers.RNN` with a ``LSTMCell`` in 2
    ways:

    - It uses ``tf.keras.layers.LSTM`` which is possibly accelerated by cuDNN on
      GPU.
    - Bidirectional outputs of each layer are reduced before feeding them to the
      next layer.
    """

    def __init__(
        self,
        num_layers,
        num_units,
        bidirectional=False,
        reducer=reducer_lib.ConcatReducer(),
        dropout=0,
        residual_connections=False,
        **kwargs
    ):
        """Initializes the layer.

        Args:
          num_layers: Number of stacked LSTM layers.
          num_units: Dimension of the output space of each LSTM.
          bidirectional: Make each layer bidirectional.
          reducer: A :class:`opennmt.layers.Reducer` instance to merge
            the bidirectional states and outputs of each layer.
          dropout: The probability to drop units in each layer output.
          residual_connections: If ``True``, each layer input will be added to its
            output.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        rnn_layers = [
            _RNNWrapper(
                tf.keras.layers.LSTM(
                    num_units, return_sequences=True, return_state=True
                ),
                bidirectional=bidirectional,
                reducer=reducer,
            )
            for _ in range(num_layers)
        ]
        self.layers = [
            common.LayerWrapper(
                layer, output_dropout=dropout, residual_connection=residual_connections
            )
            for layer in rnn_layers
        ]

    def call(self, inputs, mask=None, training=None, initial_state=None):
        all_states = []
        for i, layer in enumerate(self.layers):
            outputs, states = layer(
                inputs,
                mask=mask,
                training=training,
                initial_state=initial_state[i] if initial_state is not None else None,
            )
            all_states.append(states)
            inputs = outputs
        return outputs, tuple(all_states)


def map_v1_weights_to_cell(cell, weights):
    """Maps V1 weights to V2 RNN cell."""
    if isinstance(cell, RNNCellWrapper):
        cell = cell.cell
    if isinstance(cell, tf.keras.layers.StackedRNNCells):
        return _map_v1_weights_to_stacked_cells(cell, weights)
    elif isinstance(
        cell, (tf.keras.layers.LSTMCell, tf.compat.v1.keras.layers.LSTMCell)
    ):
        return _map_v1_weights_to_lstmcell(cell, weights)
    else:
        raise ValueError("Cannot restore V1 weights for cell %s" % str(cell))


def _map_v1_weights_to_stacked_cells(stacked_cells, weights):
    weights = weights["multi_rnn_cell"]
    m = []
    for i, cell in enumerate(stacked_cells.cells):
        m += map_v1_weights_to_cell(cell, weights["cell_%d" % i])
    return m


def _map_v1_weights_to_lstmcell(cell, weights):
    weights = weights["lstm_cell"]

    def _upgrade_weight(weight):
        is_bias = len(weight.shape) == 1
        i, j, f, o = np.split(weight, 4, axis=-1)
        if (
            is_bias
        ):  # Add forget_bias which is part of the LSTM formula in TensorFlow 1.
            f += 1
        return np.concatenate((i, f, j, o), axis=-1)  # Swap 2nd and 3rd projection.

    def _split_kernel(index):
        # TensorFlow 1 had a single kernel of shape [input_dim + units, 4 * units],
        # but TensorFlow 2 splits it into "kernel" and "recurrent_kernel".
        return tf.nest.map_structure(
            lambda w: np.split(w, [w.shape[0] - cell.units])[index], weights["kernel"]
        )

    weights = tf.nest.map_structure(_upgrade_weight, weights)
    m = []
    m.append((cell.kernel, _split_kernel(0)))
    m.append((cell.recurrent_kernel, _split_kernel(1)))
    if cell.use_bias:
        m.append((cell.bias, weights["bias"]))
    return m
