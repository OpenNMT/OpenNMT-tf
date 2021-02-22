"""Define layers related to the Google's Transformer model."""

import enum

import tensorflow as tf
import numpy as np

from opennmt.layers import common
from opennmt.utils import misc


def _lower_triangle_mask(sequence_length, maximum_length=None, dtype=tf.bool):
    batch_size = tf.shape(sequence_length)[0]
    if maximum_length is None:
        maximum_length = tf.reduce_max(sequence_length)
    mask = tf.ones([batch_size, maximum_length, maximum_length], dtype=dtype)
    mask = tf.linalg.band_part(mask, -1, 0)
    return mask


def future_mask(sequence_length, maximum_length=None, dtype=tf.bool):
    """Builds the dot product mask for future positions.

    Args:
      sequence_length: The sequence length.
      maximum_length: Optional size of the returned time dimension. Otherwise
        it is the maximum of :obj:`sequence_length`.
      dtype: The type of the mask tensor.

    Returns:
      A ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[batch_size, max_length, max_length]``.
    """
    sequence_mask = tf.sequence_mask(
        sequence_length, maxlen=maximum_length, dtype=dtype
    )
    sequence_mask = tf.expand_dims(sequence_mask, axis=1)
    mask = _lower_triangle_mask(
        sequence_length, maximum_length=maximum_length, dtype=dtype
    )
    if dtype is tf.bool:
        return tf.math.logical_and(mask, sequence_mask)
    else:
        return mask * sequence_mask


def split_heads(inputs, num_heads):
    """Splits a tensor in depth.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
      num_heads: The number of heads :math:`H`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
    """
    shape = misc.shape_list(inputs)
    outputs = tf.reshape(inputs, [shape[0], shape[1], num_heads, shape[2] // num_heads])
    outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
    return outputs


def combine_heads(inputs):
    """Concatenates heads.

    Args:
      inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.

    Returns:
      A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
    """
    shape = misc.shape_list(inputs)
    outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
    outputs = tf.reshape(outputs, [shape[0], shape[2], shape[1] * shape[3]])
    return outputs


def relative_positions(length, maximum_position, with_cache=False):
    """Builds the relative positions.

    Args:
      length: The maximum length of the sequence.
      maximum_position: The maximum relative position to represent.
      with_cache: Set to ``True`` if this function is called from a multi-head
        attention layer with cache states.

    Returns:
      Positive relative positions with shape :math:`[T or 1, T]`.
    """
    if with_cache:
        distance = tf.expand_dims(tf.range(-length + 1, 1, delta=1), 0)
    else:
        arange = tf.range(length)
        distance = tf.expand_dims(arange, 0) - tf.expand_dims(
            arange, 1
        )  # Distance to the diagonal.
    distance = tf.clip_by_value(distance, -maximum_position, maximum_position)
    return distance + maximum_position  # Return positive indices.


def matmul_with_relative_representations(a, b, transpose_b=False):
    """Multiplies :obj:`a` with the relative representations :obj:`b`.

    Args:
      a: Tensor with shape :math:`[B, H, T, _]`.
      b: Tensor with shape :math:`[T, T, _]`.

    Returns:
      Tensor with shape :math:`[B, H, T, T]`.
    """
    batch, head, time, depth = misc.shape_list(a)
    a = tf.transpose(a, perm=[2, 0, 1, 3])
    a = tf.reshape(a, [time, batch * head, depth])
    c = tf.matmul(a, b, transpose_b=transpose_b)
    c = tf.reshape(c, [time, batch, head, c.shape[-1] or tf.shape(c)[-1]])
    c = tf.transpose(c, perm=[1, 2, 0, 3])
    return c


class FeedForwardNetwork(tf.keras.layers.Layer):
    """Implements the Transformer's "Feed Forward" layer.

    .. math::

        ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2
    """

    def __init__(
        self, inner_dim, output_dim, dropout=0.1, activation=tf.nn.relu, **kwargs
    ):
        """Initializes this layer.

        Args:
          inner_dim: The number of units of the inner linear transformation.
          output_dim: The number of units of the ouput linear transformation.
          dropout: The probability to drop units from the activation output.
          activation: The activation function to apply between the two linear
            transformations.
          kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.inner = common.Dense(inner_dim, activation=activation)
        self.outer = common.Dense(output_dim)
        self.dropout = dropout

    def call(self, inputs, training=None):
        """Runs the layer."""
        inner = self.inner(inputs)
        inner = common.dropout(inner, self.dropout, training=training)
        return self.outer(inner)

    def map_v1_weights(self, weights):
        # V1 used conv1d layers that have a leading dimensions.
        weights = tf.nest.map_structure(np.squeeze, weights)
        m = []
        m += self.inner.map_v1_weights(weights["conv1d"])
        m += self.outer.map_v1_weights(weights["conv1d_1"])
        return m


class MultiHeadAttentionReduction(enum.Enum):
    """Enumeration defining how to reduce multi-head attention matrices into a
    single attention matrix.

    Possible values are:

    * ``NONE``: do not reduce and return all attention heads.
    * ``FIRST_HEAD_LAST_LAYER``: take the first attention head of the last layer.
    * ``AVERAGE_LAST_LAYER``: average of all attention heads of the last layer.
    * ``AVERAGE_ALL_LAYERS``: average of all attention heads.
    """

    NONE = 0
    FIRST_HEAD_LAST_LAYER = 1
    AVERAGE_LAST_LAYER = 2
    AVERAGE_ALL_LAYERS = 3

    @staticmethod
    def reduce(attention, policy):
        """Reduces a list of multi-head attention matrices into a single
        attention matrix.

        Args:
          attention: A list of multi-head attention matrices, with shape
            :math:`[B, H, T_t, T_s]`.
          policy: A :class:`opennmt.layers.MultiHeadAttentionReduction` value.

        Returns:
          A single attention matrix with shape :math:`[B, T_t, T_s]`.
        """
        if not isinstance(attention, list):
            raise ValueError(
                "attention should be a list of attention tensors, one per layer"
            )
        if policy == MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER:
            attention = attention[-1][:, 0]
        elif policy == MultiHeadAttentionReduction.AVERAGE_LAST_LAYER:
            attention = tf.math.reduce_mean(attention[-1], axis=1)
        elif policy == MultiHeadAttentionReduction.AVERAGE_ALL_LAYERS:
            attention = tf.concat(attention, axis=1)
            attention = tf.math.reduce_mean(attention, axis=1)
        return attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """Computes the multi-head attention as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        num_heads,
        num_units,
        dropout=0.1,
        return_attention=False,
        maximum_relative_position=None,
        **kwargs
    ):
        """Initializes this layer.

        Args:
          num_heads: The number of attention heads.
          num_units: The number of hidden units.
          dropout: The probability to drop units from the inputs.
          return_attention: If ``True``, also return the attention weights.
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        if num_units % num_heads != 0:
            raise ValueError(
                "Multi head attention requires that num_units is a"
                " multiple of %s" % num_heads
            )
        self.num_heads = num_heads
        self.num_units_per_head = num_units // num_heads
        self.linear_queries = common.Dense(num_units)
        self.linear_keys = common.Dense(num_units)
        self.linear_values = common.Dense(num_units)
        self.linear_output = common.Dense(num_units)
        self.dropout = dropout
        self.return_attention = return_attention
        self.maximum_relative_position = maximum_relative_position

    def map_v1_weights(self, weights):
        # V1 used conv1d layers that have a leading dimensions.
        weights = tf.nest.map_structure(np.squeeze, weights)

        # V1 used fused linear projections, so the weights should be split accordingly.
        def _partial_weights(key, num_splits, index):
            return tf.nest.map_structure(
                lambda w: np.split(w, num_splits, axis=0 if w.ndim == 1 else 1)[index],
                weights[key],
            )

        m = []
        if "conv1d_2" not in weights:  # Case self-attention.
            m += self.linear_queries.map_v1_weights(_partial_weights("conv1d", 3, 0))
            m += self.linear_keys.map_v1_weights(_partial_weights("conv1d", 3, 1))
            m += self.linear_values.map_v1_weights(_partial_weights("conv1d", 3, 2))
            m += self.linear_output.map_v1_weights(weights["conv1d_1"])
        else:
            m += self.linear_queries.map_v1_weights(weights["conv1d"])
            m += self.linear_keys.map_v1_weights(_partial_weights("conv1d_1", 2, 0))
            m += self.linear_values.map_v1_weights(_partial_weights("conv1d_1", 2, 1))
            m += self.linear_output.map_v1_weights(weights["conv1d_2"])
        return m

    def build(self, input_shape):
        if self.maximum_relative_position is not None:
            relative_window_size = self.maximum_relative_position * 2 + 1
            relative_repr_shape = [relative_window_size, self.num_units_per_head]
            self.relative_position_keys = self.add_weight(
                name="relative_position_keys", shape=relative_repr_shape
            )
            self.relative_position_values = self.add_weight(
                name="relative_position_values", shape=relative_repr_shape
            )
        super().build(input_shape)

    def call(self, inputs, memory=None, mask=None, cache=None, training=None):
        """Runs the layer.

        Args:
          inputs: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
          memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
            If ``None``, computes self-attention.
          mask: The dot product mask. A boolean tensor of shape :math:`[B, T_2]` or
            :math:`[B, T_1, T_2]`.
          cache: An optional tuple containing projected keys and values from the
            previous step. Tensors of shape :math:`[B, H, T_2, D / H]`.
          training: Run in training mode.

        Returns:
          A tuple with the attention context, the updated cache and the attention
          probabilities of the first head (if :obj:`return_attention` is ``True``).
        """

        def _compute_kv(x):
            keys = self.linear_keys(x)
            keys = split_heads(keys, self.num_heads)
            values = self.linear_values(x)
            values = split_heads(values, self.num_heads)
            return keys, values

        # Compute queries.
        queries = self.linear_queries(inputs)
        queries = split_heads(queries, self.num_heads)
        queries *= self.num_units_per_head ** -0.5

        # Compute keys and values.
        if memory is None:
            keys, values = _compute_kv(inputs)
            if cache:
                keys = tf.concat([cache[0], keys], axis=2)
                values = tf.concat([cache[1], values], axis=2)
        else:
            if cache:
                keys, values = tf.cond(
                    tf.equal(tf.shape(cache[0])[2], 0),
                    true_fn=lambda: _compute_kv(memory),
                    false_fn=lambda: cache,
                )
            else:
                keys, values = _compute_kv(memory)

        if self.maximum_relative_position is not None:
            if memory is not None:
                raise ValueError(
                    "Relative position representations only supports self-attention"
                )
            keys_length = tf.shape(keys)[2]
            relative_pos = relative_positions(
                keys_length, self.maximum_relative_position, with_cache=bool(cache)
            )
            relative_repr_keys = tf.nn.embedding_lookup(
                self.relative_position_keys, relative_pos
            )
            relative_repr_values = tf.nn.embedding_lookup(
                self.relative_position_values, relative_pos
            )
        else:
            relative_repr_keys = None
            relative_repr_values = None

        cache = (keys, values)

        # Dot product attention.
        dot = tf.matmul(queries, keys, transpose_b=True)
        if relative_repr_keys is not None:
            dot += matmul_with_relative_representations(
                queries, relative_repr_keys, transpose_b=True
            )
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
            mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
            dot = tf.cast(
                tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min),
                dot.dtype,
            )
        attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
        drop_attn = common.dropout(attn, self.dropout, training=training)
        heads = tf.matmul(drop_attn, values)
        if relative_repr_values is not None:
            heads += matmul_with_relative_representations(
                drop_attn, relative_repr_values
            )

        # Concatenate all heads output.
        combined = combine_heads(heads)
        outputs = self.linear_output(combined)
        if self.return_attention:
            return outputs, cache, attn
        return outputs, cache


class TransformerLayerWrapper(common.LayerWrapper):
    """Layer wrapper that applies a standard Transformer preprocessing and
    postprocessing.

    With ``pre_norm=True``:

    .. code-block:: text

        y = layer_norm(x)
        y = dropout(layer(y)) + x

    With ``pre_norm=False``:

    .. code-block:: text

        y = dropout(layer(x)) + x
        y = layer_norm(y)
    """

    def __init__(self, layer, output_dropout, pre_norm=True, **kwargs):
        """Initializes the wrapper.

        Args:
          layer: The Transformer layer to wrap.
          output_dropout: The dropout to apply on the layer output.
          pre_norm: If ``True``, layer normalization is applied before calling
            the layer. Otherwise it is applied after.
          **kwargs: Additional layer arguments.
        """
        super().__init__(
            layer,
            normalize_input=pre_norm,
            normalize_output=not pre_norm,
            output_dropout=output_dropout,
            residual_connection=True,
            **kwargs,
        )

    def map_v1_weights(self, weights):
        m = []
        m += self.input_layer_norm.map_v1_weights(weights["LayerNorm"])
        m += self.layer.map_v1_weights(weights)
        return m


class SelfAttentionEncoderLayer(tf.keras.layers.Layer):
    """Implements one self-attention encoding layer."""

    def __init__(
        self,
        num_units,
        num_heads,
        ffn_inner_dim,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        maximum_relative_position=None,
        pre_norm=True,
        **kwargs
    ):
        """Initializes the layer.

        Args:
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
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          pre_norm: If ``True``, layer normalization is applied before each
            sub-layer. Otherwise it is applied after.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = TransformerLayerWrapper(
            self.self_attention, dropout, pre_norm=pre_norm
        )
        self.ffn = FeedForwardNetwork(
            ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation
        )
        self.ffn = TransformerLayerWrapper(self.ffn, dropout, pre_norm=pre_norm)

    def call(self, x, mask=None, training=None):
        """Runs the encoder layer."""
        y, _ = self.self_attention(x, mask=mask, training=training)
        y = self.ffn(y, training=training)
        return y

    def map_v1_weights(self, weights):
        m = []
        m += self.self_attention.map_v1_weights(weights["multi_head"])
        m += self.ffn.map_v1_weights(weights["ffn"])
        return m


class SelfAttentionDecoderLayer(tf.keras.layers.Layer):
    """Implements one self-attention decoding layer."""

    def __init__(
        self,
        num_units,
        num_heads,
        ffn_inner_dim,
        num_sources=1,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        maximum_relative_position=None,
        pre_norm=True,
        **kwargs
    ):
        """Initializes the layer.

        Args:
          num_units: The number of hidden units.
          num_heads: The number of heads in the multi-head attention.
          ffn_inner_dim: The number of units of the inner linear transformation
            in the feed forward layer.
          num_sources: The number of source contexts.
          dropout: The probability to drop units from the outputs.
          attention_dropout: The probability to drop units from the attention.
          ffn_dropout: The probability to drop units from the activation output in
            the feed forward layer.
          ffn_activation: The activation function to apply between the two linear
            transformations of the feed forward layer.
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          pre_norm: If ``True``, layer normalization is applied before each
            sub-layer. Otherwise it is applied after.
          **kwargs: Additional layer arguments.
        """
        super().__init__(**kwargs)
        self.self_attention = MultiHeadAttention(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position,
        )
        self.self_attention = TransformerLayerWrapper(
            self.self_attention, dropout, pre_norm=pre_norm
        )
        self.attention = []
        for _ in range(num_sources):
            attention = MultiHeadAttention(
                num_heads, num_units, dropout=attention_dropout, return_attention=True
            )
            attention = TransformerLayerWrapper(attention, dropout, pre_norm=pre_norm)
            self.attention.append(attention)
        self.ffn = FeedForwardNetwork(
            ffn_inner_dim, num_units, dropout=ffn_dropout, activation=ffn_activation
        )
        self.ffn = TransformerLayerWrapper(self.ffn, dropout, pre_norm=pre_norm)

    def map_v1_weights(self, weights):
        m = []
        m += self.self_attention.map_v1_weights(weights["masked_multi_head"])
        m += self.attention[0].map_v1_weights(weights["multi_head"])
        m += self.ffn.map_v1_weights(weights["ffn"])
        return m

    def call(
        self,
        inputs,
        mask=None,
        memory=None,
        memory_mask=None,
        cache=None,
        training=None,
    ):
        """Runs the decoder layer."""
        if cache is None:
            cache = {}

        outputs, self_kv = self.self_attention(
            inputs, mask=mask, cache=cache.get("self_kv"), training=training
        )

        attention = []
        memory_kv = []
        if memory is not None:
            memory_cache = cache.get("memory_kv")
            if memory_cache is None:
                memory_cache = [None] * len(self.attention)
            for layer, mem, mem_mask, mem_cache in zip(
                self.attention, memory, memory_mask, memory_cache
            ):
                outputs, memory_kv_i, attention_i = layer(
                    outputs,
                    memory=mem,
                    mask=mem_mask,
                    cache=mem_cache,
                    training=training,
                )
                attention.append(attention_i)
                memory_kv.append(memory_kv_i)

        outputs = self.ffn(outputs, training=training)
        cache = dict(self_kv=self_kv, memory_kv=memory_kv)
        return outputs, cache, attention
