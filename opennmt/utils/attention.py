"""Define attention functions."""

import tensorflow as tf


def scaled_dot_attention(queries,
                         keys,
                         values,
                         mode,
                         values_length=None,
                         dropout=0.1):
  """Computes the scaled dot-product attention as described
  in https://arxiv.org/abs/1706.03762.

  Args:
    queries: The sequence of queries. A tensor of shape `[B, T1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape `[B, T2, ...]`.
    values: The sequence to attend. A tensor of shape `[B, T2, ...]`.
    mode: A `tf.estimator.ModeKeys` mode.
    values_length: The length of the values to attend.
    dropout: The probability to drop units from the inputs.

  Returns:
    The attention context.
  """
  # Scaled dot-product between queries and keys.
  dot = tf.matmul(queries, keys, transpose_b=True)
  dot = tf.div(dot, tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32)))

  if not values_length is None:
    # Give no weight to illegal connections.
    mask = tf.sequence_mask(
      values_length,
      tf.shape(values)[1],
      dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=1)
    dot = dot * mask + ((1.0 - mask) * tf.float32.min)

  # Compute attention weights.
  attn = tf.nn.softmax(dot)

  attn = tf.contrib.layers.dropout(
    attn,
    keep_prob=1.0 - dropout,
    is_training=mode == tf.estimator.ModeKeys.TRAIN)

  # Compute attention context.
  context = tf.matmul(attn, values)

  return context


def multi_head_attention(num_heads,
                         queries,
                         keys,
                         values,
                         mode,
                         values_length=None,
                         dropout=0.1):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.

  Args:
    num_heads: The number of attention heads.
    queries: The sequence of queries. A tensor of shape `[B, T1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape `[B, T2, ...]`.
    values: The sequence to attend. A tensor of shape `[B, T2, ...]`.
    mode: A `tf.estimator.ModeKeys` mode.
    values_length: The length of the values to attend.
    dropout: The probability to drop units from the inputs.

  Returns:
    The concatenated attention context of each head.
  """
  heads = []

  input_dim = keys.get_shape().as_list()[-1]
  head_dim = input_dim / num_heads

  with tf.variable_scope("multi_head"):
    for i in range(num_heads):
      with tf.variable_scope("head_" + str(i)):
        # Project queries, keys and values to different and smaller subspaces.
        queries_proj = tf.layers.dense(
          queries,
          head_dim,
          use_bias=False)
        keys_proj = tf.layers.dense(
          keys,
          head_dim,
          use_bias=False)
        values_proj = tf.layers.dense(
          values,
          head_dim,
          use_bias=False)

        head_i = scaled_dot_attention(
          queries_proj,
          keys_proj,
          values_proj,
          mode,
          values_length=values_length,
          dropout=dropout)

        heads.append(head_i)

    # Concatenate all heads output.
    return tf.concat(heads, axis=2)


def masked_multi_head_attention(num_heads,
                                queries,
                                keys,
                                values,
                                mode,
                                keep_prob=0.9):
  """Computes the masked multi-head attention giving no
  weights to future timesteps as described in
  https://arxiv.org/abs/1706.03762.

  Args:
    num_heads: The number of attention heads.
    queries: The sequence of queries. A tensor of shape `[B, T, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape `[B, T, ...]`.
    values: The sequence to attend. A tensor of shape `[B, T, ...]`.
    mode: A `tf.estimator.ModeKeys` mode.
    values_length: The length of the values to attend.
    dropout: The probability to drop units from the inputs.

  Returns:
    The concatenated attention context of each head.
  """
  values_length = tf.range(tf.shape(values)[1])

  return multi_head_attention(
    num_heads,
    queries,
    keys,
    values,
    mode,
    values_length=values_length,
    dropout=dropout)
