"""Define RNN-based decoders."""

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt.decoders import decoder
from opennmt.layers import bridge
from opennmt.layers import common
from opennmt.layers import rnn
from opennmt.layers import transformer
from opennmt.layers.rnn import map_v1_weights_to_cell


class RNNDecoder(decoder.Decoder):
    """A basic RNN decoder."""

    def __init__(
        self,
        num_layers,
        num_units,
        bridge_class=None,
        cell_class=None,
        dropout=0.3,
        residual_connections=False,
        **kwargs
    ):
        """Initializes the decoder parameters.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer.
          bridge_class: A :class:`opennmt.layers.Bridge` class to pass the
            encoder state to the decoder. Default to
            :class:`opennmt.layers.ZeroBridge`.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell. Defaults to a LSTM cell.
          dropout: The probability to drop units in each layer output.
          residual_connections: If ``True``, each layer input will be added to its
            output.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.dropout = dropout
        self.cell = rnn.make_rnn_cell(
            num_layers,
            num_units,
            dropout=dropout,
            residual_connections=residual_connections,
            cell_class=cell_class,
        )
        if bridge_class is None:
            bridge_class = bridge.ZeroBridge
        self.bridge = bridge_class()

    def _get_initial_state(self, batch_size, dtype, initial_state=None):
        cell_initial_state = self.cell.get_initial_state(
            batch_size=batch_size, dtype=dtype
        )
        if initial_state is not None:
            cell_initial_state = self.bridge(initial_state, cell_initial_state)
        return cell_initial_state

    def step(
        self,
        inputs,
        timestep,
        state=None,
        memory=None,
        memory_sequence_length=None,
        training=None,
    ):
        outputs, state = self.cell(inputs, state, training=training)
        return outputs, state, None


class AttentionalRNNDecoder(RNNDecoder):
    """A RNN decoder with attention."""

    def __init__(
        self,
        num_layers,
        num_units,
        bridge_class=None,
        attention_mechanism_class=None,
        cell_class=None,
        dropout=0.3,
        residual_connections=False,
        first_layer_attention=False,
        attention_layer_activation=tf.math.tanh,
        **kwargs
    ):
        """Initializes the decoder parameters.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer.
          bridge: A :class:`opennmt.layers.Bridge` to pass the encoder state
            to the decoder.
          attention_mechanism_class: A class inheriting from
            ``tfa.seq2seq.AttentionMechanism``. Defaults to
            ``tfa.seq2seq.LuongAttention``.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell.
          dropout: The probability to drop units in each layer output.
          residual_connections: If ``True``, each layer input will be added to its
            output.
          first_layer_attention: If ``True``, output attention after the first layer.
          attention_layer_activation: The activation to produce the attentional hidden
            state. Defaults to tanh following Luong paper (equation (5) in
            https://arxiv.org/abs/1508.04025).
          **kwargs: Additional layer arguments.
        """
        super().__init__(
            num_layers,
            num_units,
            bridge_class=bridge_class,
            cell_class=cell_class,
            dropout=dropout,
            residual_connections=residual_connections,
            **kwargs,
        )
        if attention_mechanism_class is None:
            attention_mechanism_class = tfa.seq2seq.LuongAttention
        self.attention_mechanism = attention_mechanism_class(self.cell.output_size)

        def _add_attention(cell):
            # Produce Luong-style attentional hidden states.
            attention_layer = common.Dense(
                cell.output_size, use_bias=False, activation=attention_layer_activation
            )
            wrapper = tfa.seq2seq.AttentionWrapper(
                cell, self.attention_mechanism, attention_layer=attention_layer
            )
            return wrapper

        if first_layer_attention:
            self.cell.cells[0] = _add_attention(self.cell.cells[0])
        else:
            self.cell = _add_attention(self.cell)
        self.dropout = dropout
        self.first_layer_attention = first_layer_attention

    @property
    def support_alignment_history(self):
        return True

    def _get_initial_state(self, batch_size, dtype, initial_state=None):
        # Reset memory of attention mechanism.
        self.attention_mechanism.setup_memory(
            self.memory, memory_sequence_length=self.memory_sequence_length
        )
        decoder_state = self.cell.get_initial_state(batch_size=batch_size, dtype=dtype)
        if initial_state is not None:
            if self.first_layer_attention:
                cell_state = list(decoder_state)
                cell_state[0] = decoder_state[0].cell_state
                cell_state = self.bridge(initial_state, cell_state)
                cell_state[0] = decoder_state[0].clone(cell_state=cell_state[0])
                decoder_state = tuple(cell_state)
            else:
                cell_state = self.bridge(initial_state, decoder_state.cell_state)
                decoder_state = decoder_state.clone(cell_state=cell_state)
        return decoder_state

    def step(
        self,
        inputs,
        timestep,
        state=None,
        memory=None,
        memory_sequence_length=None,
        training=None,
    ):
        outputs, state = self.cell(inputs, state, training=training)
        outputs = common.dropout(outputs, self.dropout, training=training)
        if self.first_layer_attention:
            attention = state[0].alignments
        else:
            attention = state.alignments
        return outputs, state, attention

    def map_v1_weights(self, weights):
        if self.first_layer_attention or not isinstance(
            self.attention_mechanism, tfa.seq2seq.LuongAttention
        ):
            raise ValueError(
                "Can only map V1 weights for RNN decoder with Luong attention "
                "on the last layer"
            )
        m = super().map_v1_weights(weights)
        m += common.Dense.map_v1_weights(
            self.attention_mechanism.memory_layer, weights["memory_layer"]
        )
        weights = weights["decoder"]["attention_wrapper"]
        m += common.Dense.map_v1_weights(
            self.cell._attention_layers[0], weights["attention_layer"]
        )
        m += map_v1_weights_to_cell(self.cell._cell, weights)
        return m


class RNMTPlusDecoder(decoder.Decoder):
    """The RNMT+ decoder described in https://arxiv.org/abs/1804.09849."""

    def __init__(
        self, num_layers, num_units, num_heads, dropout=0.3, cell_class=None, **kwargs
    ):
        """Initializes the decoder parameters.

        Args:
          num_layers: The number of layers.
          num_units: The number of units in each layer.
          num_heads: The number of attention heads.
          dropout: The probability to drop units from the decoder input and in each
            layer output.
          cell_class: The inner cell class or a callable taking :obj:`num_units` as
            argument and returning a cell. Defaults to a layer normalized LSTM cell.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        if cell_class is None:
            cell_class = tfa.rnn.LayerNormLSTMCell
        self.num_heads = num_heads
        self.num_units = num_units
        self.dropout = dropout
        self.cells = [cell_class(num_units) for _ in range(num_layers)]
        self.multi_head_attention = transformer.MultiHeadAttention(
            num_heads, num_units, dropout=dropout, return_attention=True
        )

    @property
    def support_alignment_history(self):
        return True

    def _get_initial_state(self, batch_size, dtype, initial_state=None):
        return tuple(
            cell.get_initial_state(batch_size=batch_size, dtype=dtype)
            for cell in self.cells
        )

    def step(
        self,
        inputs,
        timestep,
        state=None,
        memory=None,
        memory_sequence_length=None,
        training=None,
    ):
        inputs = common.dropout(inputs, rate=self.dropout, training=training)

        new_states = []
        last_outputs, state_0 = self.cells[0](inputs, state[0])
        new_states.append(state_0)

        if memory_sequence_length is not None:
            memory_mask = tf.sequence_mask(
                memory_sequence_length, maxlen=tf.shape(memory)[1]
            )
        else:
            memory_mask = None

        context, _, attention = self.multi_head_attention(
            tf.expand_dims(last_outputs, 1),
            memory=memory,
            mask=memory_mask,
            training=training,
        )
        attention = attention[:, 0, 0]  # Use the first head for the attention vector.
        context = tf.squeeze(context, axis=1)

        for i in range(1, len(self.cells)):
            inputs = tf.concat([last_outputs, context], axis=-1)
            outputs, state_i = self.cells[i](inputs, state[i], training=training)
            new_states.append(state_i)
            outputs = common.dropout(outputs, rate=self.dropout, training=training)
            if i >= 2:
                outputs += last_outputs
            last_outputs = outputs

        final = tf.concat([last_outputs, context], -1)
        return final, tuple(new_states), attention
