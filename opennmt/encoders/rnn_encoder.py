"""Define RNN-based encoders."""

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt.encoders.encoder import Encoder, SequentialEncoder
from opennmt.layers.reducer import ConcatReducer, JoinReducer, pad_in_time
from opennmt.layers import common
from opennmt.layers import rnn


class _RNNEncoderBase(Encoder):
    """Base class for RNN encoders."""

    def __init__(self, rnn_layer, **kwargs):
        """Initializes the encoder.

        Args:
          rnn_layer: The RNN layer used to encode the inputs.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.rnn = rnn_layer

    def call(self, inputs, sequence_length=None, training=None):
        mask = self.build_mask(inputs, sequence_length=sequence_length)
        outputs, states = self.rnn(inputs, mask=mask, training=training)
        return outputs, states, sequence_length


class RNNEncoder(_RNNEncoderBase):
    """A RNN sequence encoder."""

    def __init__(
        self,
        num_layers,
        num_units,
        bidirectional=False,
        residual_connections=False,
        dropout=0.3,
        reducer=ConcatReducer(),
        cell_class=None,
        **kwargs
    ):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer.
          bidirectional: Use a bidirectional RNN.
          residual_connections: If ``True``, each layer input will be added to its
            output.
          dropout: The probability to drop units in each layer output.
          reducer: A :class:`opennmt.layers.Reducer` instance to merge
            bidirectional state and outputs.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell. Defaults to a LSTM cell.
          **kwargs: Additional layer arguments.
        """
        cell = rnn.make_rnn_cell(
            num_layers,
            num_units,
            dropout=dropout,
            residual_connections=residual_connections,
            cell_class=cell_class,
        )
        rnn_layer = rnn.RNN(cell, bidirectional=bidirectional, reducer=reducer)
        super().__init__(rnn_layer, **kwargs)

    def map_v1_weights(self, weights):
        return self.rnn.map_v1_weights(weights)


class LSTMEncoder(_RNNEncoderBase):
    """A LSTM sequence encoder.

    See Also:
      :class:`opennmt.layers.LSTM` for differences between this encoder and
      :class:`opennmt.encoders.RNNEncoder` with a `LSTMCell`.
    """

    def __init__(
        self,
        num_layers,
        num_units,
        bidirectional=False,
        residual_connections=False,
        dropout=0.3,
        reducer=ConcatReducer(),
        **kwargs
    ):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer output.
          bidirectional: Make each LSTM layer bidirectional.
          residual_connections: If ``True``, each layer input will be added to its
            output.
          dropout: The probability to drop units in each layer output.
          reducer: A :class:`opennmt.layers.Reducer` instance to merge
            bidirectional state and outputs.
          **kwargs: Additional layer arguments.
        """
        lstm_layer = rnn.LSTM(
            num_layers,
            num_units,
            bidirectional=bidirectional,
            reducer=reducer,
            dropout=dropout,
            residual_connections=residual_connections,
        )
        super().__init__(lstm_layer, **kwargs)


class GNMTEncoder(SequentialEncoder):
    """The RNN encoder used in GNMT as described in
    https://arxiv.org/abs/1609.08144.
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
            raise ValueError("GNMTEncoder requires at least 2 layers")
        bidirectional = LSTMEncoder(1, num_units, bidirectional=True, dropout=dropout)
        unidirectional = LSTMEncoder(
            num_layers - 1, num_units, dropout=dropout, residual_connections=True
        )
        super().__init__([bidirectional, unidirectional])


class RNMTPlusEncoder(SequentialEncoder):
    """The RNMT+ encoder described in https://arxiv.org/abs/1804.09849."""

    def __init__(self, num_layers=6, num_units=1024, cell_class=None, dropout=0.3):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each RNN layer and the final output.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell. Defaults to a layer normalized LSTM cell.
          dropout: The probability to drop units in each layer output.
        """
        if cell_class is None:
            cell_class = tfa.rnn.LayerNormLSTMCell
        layers = [
            RNNEncoder(
                1, num_units, bidirectional=True, dropout=0.0, cell_class=cell_class
            )
            for _ in range(num_layers)
        ]
        layers = [
            common.LayerWrapper(layer, output_dropout=dropout, residual_connection=True)
            for layer in layers
        ]
        super().__init__(layers)
        self.dropout = dropout
        self.projection = tf.keras.layers.Dense(num_units)

    def call(self, inputs, sequence_length=None, training=None):
        inputs = common.dropout(inputs, self.dropout, training=training)
        outputs, state, sequence_length = super().call(
            inputs, sequence_length=sequence_length, training=training
        )
        projected = self.projection(outputs)
        return (projected, state, sequence_length)


class PyramidalRNNEncoder(Encoder):
    """An encoder that reduces the time dimension after each bidirectional layer."""

    def __init__(
        self, num_layers, num_units, reduction_factor=2, cell_class=None, dropout=0.3
    ):
        """Initializes the parameters of the encoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer.
          reduction_factor: The time reduction factor.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell. Defaults to a LSTM cell.
          dropout: The probability to drop units in each layer output.
        """
        super().__init__()
        self.reduction_factor = reduction_factor
        self.state_reducer = JoinReducer()
        self.layers = [
            RNNEncoder(
                1,
                num_units // 2,
                bidirectional=True,
                reducer=ConcatReducer(),
                cell_class=cell_class,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ]

    def call(self, inputs, sequence_length=None, training=None):
        encoder_state = []

        for layer_index, layer in enumerate(self.layers):
            input_depth = inputs.shape[-1]

            if layer_index == 0:
                # For the first input, make the number of timesteps a multiple of the
                # total reduction factor.
                total_reduction_factor = pow(
                    self.reduction_factor, len(self.layers) - 1
                )

                current_length = tf.shape(inputs)[1]
                factor = tf.cast(current_length, tf.float32) / total_reduction_factor
                new_length = (
                    tf.cast(tf.math.ceil(factor), tf.int32) * total_reduction_factor
                )
                inputs = pad_in_time(inputs, new_length - current_length)

                # Lengths should not be smaller than the total reduction factor.
                sequence_length = tf.maximum(sequence_length, total_reduction_factor)
            else:
                # In other cases, reduce the time dimension.
                inputs = tf.reshape(
                    inputs,
                    [tf.shape(inputs)[0], -1, input_depth * self.reduction_factor],
                )
                if sequence_length is not None:
                    sequence_length //= self.reduction_factor

            outputs, state, sequence_length = layer(
                inputs, sequence_length=sequence_length, training=training
            )

            encoder_state.append(state)
            inputs = outputs

        return (outputs, self.state_reducer(encoder_state), sequence_length)
