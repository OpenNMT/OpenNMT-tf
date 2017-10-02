"""Define functions related to the Google's Transformer model."""

import tensorflow as tf


def scaled_dot_attention(queries,
                         keys,
                         values,
                         mode,
                         values_length=None,
                         mask_future=False,
                         dropout=0.1):
  """Computes the scaled dot-product attention as described
  in https://arxiv.org/abs/1706.03762.

  Args:
    queries: The sequence of queries. A tensor of shape `[B, T1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape `[B, T2, ...]`.
    values: The sequence to attend. A tensor of shape `[B, T2, ...]`.
    mode: A `tf.estimator.ModeKeys` mode.
    values_length: The length of the values to attend.
    mask_future: Mask attention to future positions.
    dropout: The probability to drop units from the inputs.

  Returns:
    A tuple `(context vector, attention vector)`.
  """
  # Scaled dot-product between queries and keys.
  dot = tf.matmul(queries, keys, transpose_b=True)
  dot = tf.div(dot, tf.sqrt(tf.cast(tf.shape(keys)[-1], tf.float32)))

  if values_length is not None:
    # Give no weight to illegal connections.
    if mask_future:
      # When masking the future, a position can only attend to previous timesteps.
      mask = tf.map_fn(
          lambda x: tf.sequence_mask(
              tf.minimum(tf.range(tf.shape(values)[1]) + 1, x),
              maxlen=tf.shape(values)[1],
              dtype=tf.float32),
          values_length,
          dtype=tf.float32)
    else:
      # Otherwise, simply prevent attention on out-of-range positions.
      mask = tf.sequence_mask(
          values_length,
          maxlen=tf.shape(values)[1],
          dtype=tf.float32)
      mask = tf.expand_dims(mask, axis=1)

    dot = dot * mask + ((1.0 - mask) * tf.float32.min)

  # Compute attention weights.
  attn = tf.nn.softmax(dot)
  attn = tf.layers.dropout(
      attn,
      rate=dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Compute attention context.
  context = tf.matmul(attn, values)

  return context, attn


def multi_head_attention(num_heads,
                         queries,
                         keys,
                         values,
                         mode,
                         values_length=None,
                         mask_future=False,
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
    mask_future: Mask attention to future positions.
    dropout: The probability to drop units from the inputs.

  Returns:
    The concatenated attention context of each head.
  """
  input_dim = keys.get_shape().as_list()[-1]

  if input_dim % num_heads != 0:
    raise ValueError("Multi head attention requires the input dimension to be a"
                     " multiple of {}".format(num_heads))

  head_dim = input_dim / num_heads
  heads = []

  for i in range(num_heads):
    with tf.variable_scope("head_{}".format(i)):
      # Project queries, keys and values to different and smaller subspaces.
      queries_proj = tf.layers.conv1d(queries, head_dim, 1)
      keys_proj = tf.layers.conv1d(keys, head_dim, 1)
      values_proj = tf.layers.conv1d(values, head_dim, 1)

      head_i, _ = scaled_dot_attention(
          queries_proj,
          keys_proj,
          values_proj,
          mode,
          values_length=values_length,
          mask_future=mask_future,
          dropout=dropout)

      heads.append(head_i)

  # Concatenate all heads output.
  combined = tf.concat(heads, axis=2)
  outputs = tf.layers.conv1d(combined, input_dim, 1)

  return outputs

def feed_forward(x, inner_dim):
  """Implements the Transformer's "Feed Forward" layer.

  ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2

  Args:
    x: The input.
    inner_dim: The number of units of the inner linear transformation.

  Returns:
    The transformed input.
  """
  input_dim = x.get_shape().as_list()[-1]

  inner = tf.layers.conv1d(x, inner_dim, 1, activation=tf.nn.relu)
  outer = tf.layers.conv1d(inner, input_dim, 1)

  return outer

def add_and_norm(inputs,
                 outputs,
                 mode,
                 dropout=0.1):
  """Implements the Transformer's "Add & Norm" layer.

  Args:
    inputs: The input of the previous layer.
    outputs: The output of the previous layer.
    mode: A `tf.estimator.ModeKeys` mode.
    dropout: The probability to drop units in `outputs`.

  Returns:
    The residual and normalized output.
  """
  outputs = tf.layers.dropout(
      outputs,
      rate=dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)
  outputs += inputs
  outputs = tf.contrib.layers.layer_norm(outputs)
  return outputs
