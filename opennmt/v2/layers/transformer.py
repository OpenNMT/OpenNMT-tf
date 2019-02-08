# pylint: disable=arguments-differ

"""Define layers related to the Google's Transformer model."""

import tensorflow as tf

from opennmt.layers.transformer import combine_heads, split_heads


class FeedForwardNetwork(tf.keras.layers.Layer):
  """Implements the Transformer's "Feed Forward" layer.

  .. math::

      ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2
  """

  def __init__(self, inner_dim, output_dim, dropout=0.1, **kwargs):
    """Initializes this layer.

    Args:
      inner_dim: The number of units of the inner linear transformation.
      output_dim: The number of units of the ouput linear transformation.
      dropout: The probability to drop units from the inner transformation.
      kwargs: Additional layer arguments.
    """
    super(FeedForwardNetwork, self).__init__(**kwargs)
    self.inner = tf.keras.layers.Dense(inner_dim, activation=tf.nn.relu)
    self.outer = tf.keras.layers.Dense(output_dim)
    self.dropout = dropout

  def call(self, inputs, training=None):
    """Runs the layer."""
    inner = self.inner(inputs)
    if training:
      inner = tf.nn.dropout(inner, rate=self.dropout)
    return self.outer(inner)


class MultiHeadAttention(tf.keras.layers.Layer):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               num_heads,
               num_units,
               dropout=0.1,
               return_attention=False,
               **kwargs):
    """Initializes this layers.

    Args:
      num_heads: The number of attention heads.
      num_units: The number of hidden units.
      dropout: The probability to drop units from the inputs.
      return_attention: If ``True``, also return the attention weights of the
        first head.
      kwargs: Additional layer arguments.
    """
    super(MultiHeadAttention, self).__init__(**kwargs)
    if num_units % num_heads != 0:
      raise ValueError("Multi head attention requires that num_units is a"
                       " multiple of %s" % num_heads)
    self.num_heads = num_heads
    self.num_units = num_units
    self.linear_queries = tf.keras.layers.Dense(num_units)
    self.linear_keys = tf.keras.layers.Dense(num_units)
    self.linear_values = tf.keras.layers.Dense(num_units)
    self.linear_output = tf.keras.layers.Dense(num_units)
    self.dropout = dropout
    self.return_attention = return_attention

  def call(self, inputs, memory=None, mask=None, cache=None, training=None):
    """Runs the layer.

    Args:
      inputs: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
      memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
        If ``None``, computes self-attention.
      mask: A ``tf.Tensor`` applied to the dot product.
      cache: A dictionary containing pre-projected keys and values.
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
    queries *= (self.num_units // self.num_heads)**-0.5

    # Compute keys and values.
    if memory is None:
      keys, values = _compute_kv(inputs)
      if cache is not None:
        keys = tf.concat([cache[0], keys], axis=2)
        values = tf.concat([cache[1], values], axis=2)
    else:
      if not cache or tf.equal(tf.shape(cache[0])[2], 0):
        keys, values = _compute_kv(memory)
      else:
        keys, values = cache
    cache = (keys, values)

    # Dot product attention.
    dot = tf.matmul(queries, keys, transpose_b=True)
    if mask is not None:
      mask = tf.expand_dims(tf.cast(mask, tf.float32), 1)  # Broadcast on heads dimension.
      dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)
    attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
    drop_attn = tf.nn.dropout(attn, dropout) if training else attn
    heads = tf.matmul(drop_attn, values)

    # Concatenate all heads output.
    combined = combine_heads(heads)
    outputs = self.linear_output(combined)
    if self.return_attention:
      return outputs, cache, attn
    return outputs, cache
