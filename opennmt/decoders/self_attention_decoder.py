"""Define self-attention decoder."""

import tensorflow as tf

from opennmt.decoders import decoder
from opennmt.layers import common, transformer
from opennmt.layers.position import SinusoidalPositionEncoder


class SelfAttentionDecoder(decoder.Decoder):
    """Encoder using self-attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        num_layers,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        num_sources=1,
        maximum_relative_position=None,
        attention_reduction=transformer.MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER,
        pre_norm=True,
        **kwargs
    ):
        """Initializes the parameters of the decoder.

        Args:
          num_layers: The number of layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          ffn_dropout: The probability to drop units from the activation output in
            the feed forward layer.
          ffn_activation: The activation function to apply between the two linear
            transformations of the feed forward layer.
          position_encoder_class: The :class:`opennmt.layers.PositionEncoder`
            class to use for position encoding (or a callable that returns an
            instance).
          num_sources: The number of source contexts expected by this decoder.
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          attention_reduction: A :class:`opennmt.layers.MultiHeadAttentionReduction`
            value to specify how to reduce multi-head attention matrices.
          pre_norm: If ``True``, layer normalization is applied before each
            sub-layer. Otherwise it is applied after.
          **kwargs: Additional layer arguments.
        """
        super().__init__(num_sources=num_sources, **kwargs)
        self.num_units = num_units
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_reduction = attention_reduction
        self.position_encoder = None
        if position_encoder_class is not None:
            self.position_encoder = position_encoder_class()
        self.layer_norm = common.LayerNorm() if pre_norm else None
        self.layers = [
            transformer.SelfAttentionDecoderLayer(
                self.num_units,
                self.num_heads,
                ffn_inner_dim,
                num_sources=num_sources,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
            )
            for i in range(num_layers)
        ]

    @property
    def minimum_sources(self):
        return 0

    @property
    def maximum_sources(self):
        return 1e6  # An arbitrary large number.

    @property
    def support_alignment_history(self):
        return True

    def map_v1_weights(self, weights):
        m = super().map_v1_weights(weights)
        m += self.layer_norm.map_v1_weights(weights["LayerNorm"])
        for i, layer in enumerate(self.layers):
            m += layer.map_v1_weights(weights["layer_%d" % i])
        return m

    def _run(
        self,
        inputs,
        sequence_length=None,
        cache=None,
        memory=None,
        memory_sequence_length=None,
        step=None,
        training=None,
    ):
        # Process inputs.
        inputs *= self.num_units ** 0.5
        if self.position_encoder is not None:
            inputs = self.position_encoder(
                inputs, position=step + 1 if step is not None else None
            )
        inputs = common.dropout(inputs, self.dropout, training=training)

        # Prepare query mask.
        mask = None
        if step is None:
            maximum_length = tf.shape(inputs)[1]
            if sequence_length is None:
                batch_size = tf.shape(inputs)[0]
                sequence_length = tf.fill([batch_size], maximum_length)
            mask = transformer.future_mask(
                sequence_length, maximum_length=maximum_length
            )

        # Prepare memory mask.
        memory_mask = None
        if memory is not None:
            if not isinstance(memory, (list, tuple)):
                memory = (memory,)
        if memory_sequence_length is not None:
            if not isinstance(memory_sequence_length, (list, tuple)):
                memory_sequence_length = (memory_sequence_length,)
            memory_mask = [
                tf.sequence_mask(mem_length, maxlen=tf.shape(mem)[1])
                for mem, mem_length in zip(memory, memory_sequence_length)
            ]

        # Run each layer.
        new_cache = []
        attention = []
        for i, layer in enumerate(self.layers):
            inputs, layer_cache, layer_attention = layer(
                inputs,
                mask=mask,
                memory=memory,
                memory_mask=memory_mask,
                cache=cache[i] if cache is not None else None,
                training=training,
            )
            attention.append(layer_attention)
            new_cache.append(layer_cache)
        outputs = self.layer_norm(inputs) if self.layer_norm is not None else inputs

        # Convert list of shape num_layers x num_sources to num_sources x num_layers
        attention = list(map(list, zip(*attention)))
        if attention:
            attention = transformer.MultiHeadAttentionReduction.reduce(
                attention[0],  # Get attention to the first source.
                self.attention_reduction,
            )
        else:
            attention = None

        return outputs, new_cache, attention

    def forward(
        self,
        inputs,
        sequence_length=None,
        initial_state=None,
        memory=None,
        memory_sequence_length=None,
        input_fn=None,
        sampling_probability=None,
        training=None,
    ):
        _ = initial_state
        _ = input_fn
        if sampling_probability is not None:
            raise ValueError("Scheduled sampling is not supported by this decoder")
        outputs, state, attention = self._run(
            inputs,
            sequence_length=sequence_length,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            training=training,
        )
        logits = self.output_layer(outputs)
        return logits, state, attention

    def step(
        self,
        inputs,
        timestep,
        state=None,
        memory=None,
        memory_sequence_length=None,
        training=None,
    ):
        inputs = tf.expand_dims(inputs, 1)
        outputs, state, attention = self._run(
            inputs,
            cache=state,
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            step=timestep,
            training=training,
        )
        outputs = tf.squeeze(outputs, axis=1)
        if attention is not None:
            attention = tf.squeeze(attention, axis=1)
        return outputs, state, attention

    def _get_initial_state(self, batch_size, dtype, initial_state=None):
        # The decoder state contains the keys and values projections of the previous timesteps.
        _ = initial_state
        cache = []
        for _ in self.layers:
            shape = [batch_size, self.num_heads, 0, self.num_units // self.num_heads]
            self_kv = (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
            memory_kv = [
                (tf.zeros(shape, dtype=dtype), tf.zeros(shape, dtype=dtype))
                for _ in range(self.num_sources)
            ]
            cache.append(dict(self_kv=self_kv, memory_kv=memory_kv))
        return cache

    def _get_state_reorder_flags(self):
        # We don't need to reorder memory_kv as it is the same for all beams.
        return [
            {
                "self_kv": (True, True),
                "memory_kv": [(False, False) for _ in range(self.num_sources)],
            }
            for _ in self.layers
        ]
