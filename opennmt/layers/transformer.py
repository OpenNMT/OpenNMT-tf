"""Define layers related to the Google's Transformer model."""

import tensorflow as tf


def tile_sequence_length(sequence_length, num_heads):
  """Tiles lengths :obj:`num_heads` times.

  Args:
    sequence_length: The sequence length.
    num_heads: The number of heads.

  Returns:
    A ``tf.Tensor`` where each length is replicated :obj:`num_heads` times.
  """
  sequence_length = tf.tile(sequence_length, [num_heads])
  sequence_length = tf.reshape(sequence_length, [num_heads, -1])
  sequence_length = tf.transpose(sequence_length, perm=[1, 0])
  sequence_length = tf.reshape(sequence_length, [-1])
  return sequence_length

def build_sequence_mask(sequence_length,
                        num_heads=None,
                        maximum_length=None,
                        dtype=tf.float32):
  """Builds the dot product mask.

  Args:
    sequence_length: The sequence length.
    num_heads: The number of heads.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A broadcastable ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, num_heads, 1, max_length]``.
  """
  if num_heads is not None:
    sequence_length = tile_sequence_length(sequence_length, num_heads)
  mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = tf.expand_dims(mask, axis=1)
  if num_heads is not None:
    mask = tf.reshape(mask, [-1, num_heads, tf.shape(mask)[1], tf.shape(mask)[2]])
  return mask

def _lower_triangle_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  batch_size = tf.shape(sequence_length)[0]
  if maximum_length is None:
    maximum_length = tf.reduce_max(sequence_length)
  mask = tf.ones([batch_size, maximum_length, maximum_length], dtype=dtype)
  mask = tf.matrix_band_part(mask, -1, 0)
  return mask

def build_future_mask(sequence_length,
                      num_heads=None,
                      maximum_length=None,
                      dtype=tf.float32):
  """Builds the dot product mask for future positions.

  Args:
    sequence_length: The sequence length.
    num_heads: The number of heads.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, num_heads, max_length, max_length]``.
  """
  if num_heads is not None:
    sequence_length = tile_sequence_length(sequence_length, num_heads)
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  mask *= tf.expand_dims(sequence_mask, axis=1)
  if num_heads is not None:
    mask = tf.reshape(mask, [-1, num_heads, tf.shape(mask)[1], tf.shape(mask)[2]])
  return mask

def cumulative_average_mask(sequence_length, maximum_length=None, dtype=tf.float32):
  """Builds the mask to compute the cumulative average as described in
  https://arxiv.org/abs/1805.00631.

  Args:
    sequence_length: The sequence length.
    maximum_length: Optional size of the returned time dimension. Otherwise
      it is the maximum of :obj:`sequence_length`.
    dtype: The type of the mask tensor.

  Returns:
    A ``tf.Tensor`` of type :obj:`dtype` and shape
    ``[batch_size, max_length, max_length]``.
  """
  sequence_mask = tf.sequence_mask(sequence_length, maxlen=maximum_length, dtype=dtype)
  mask = _lower_triangle_mask(sequence_length, maximum_length=maximum_length, dtype=dtype)
  mask *= tf.expand_dims(sequence_mask, axis=2)
  weight = tf.range(1, tf.cast(tf.shape(mask)[1] + 1, dtype), dtype=dtype)
  mask /= tf.expand_dims(weight, 1)
  return mask

def cumulative_average(inputs, mask_or_step, cache=None):
  """Computes the cumulative average as described in
  https://arxiv.org/abs/1805.00631.

  Args:
    inputs: The sequence to average. A tensor of shape :math:`[B, T, D]`.
    mask_or_step: If :obj:`cache` is set, this is assumed to be the current step
      of the dynamic decoding. Otherwise, it is the mask matrix used to compute
      the cumulative average.
    cache: A dictionnary containing the cumulative average of the previous step.

  Returns:
    The cumulative average, a tensor of the same shape and type as :obj:`inputs`.
  """
  if cache is not None:
    step = tf.cast(mask_or_step, inputs.dtype)
    aa = (inputs + step * cache["prev_g"]) / (step + 1.0)
    cache["prev_g"] = aa
    return aa
  else:
    mask = mask_or_step
    return tf.matmul(mask, inputs)

def fused_projection(inputs, num_units, num_outputs=1):
  """Projects the same input into multiple output spaces.

  Args:
    inputs: The inputs to project.
    num_units: The number of output units of each space.
    num_outputs: The number of output spaces.

  Returns:
    :obj:`num_outputs` ``tf.Tensor`` of depth :obj:`num_units`.
  """
  return tf.split(
      tf.layers.conv1d(inputs, num_units * num_outputs, 1), num_outputs, axis=2)

def split_heads(inputs, num_heads):
  """Splits a tensor in depth.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, T, D]`.
    num_heads: The number of heads :math:`H`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, H, T, D / H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  outputs = tf.reshape(
      inputs, [tf.shape(inputs)[0], tf.shape(inputs)[1], num_heads, depth // num_heads])
  outputs = tf.transpose(outputs, perm=[0, 2, 1, 3])
  return outputs

def combine_heads(inputs):
  """Concatenates heads.

  Args:
    inputs: A ``tf.Tensor`` of shape :math:`[B, H, T, D]`.

  Returns:
    A ``tf.Tensor`` of shape :math:`[B, T, D * H]`.
  """
  static_shape = inputs.get_shape().as_list()
  depth = static_shape[-1]
  num_heads = static_shape[1]
  outputs = tf.transpose(inputs, perm=[0, 2, 1, 3])
  outputs = tf.reshape(outputs, [tf.shape(outputs)[0], tf.shape(outputs)[1], depth * num_heads])
  return outputs

def dot_product_attention(queries,
                          keys,
                          values,
                          mode,
                          mask=None,
                          dropout=0.0):
  """Computes the dot product attention.

  Args:
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    keys: The sequence use to calculate attention scores. A tensor of shape
      :math:`[B, T_2, ...]`.
    values: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
    mode: A ``tf.estimator.ModeKeys`` mode.
    mask: A ``tf.Tensor`` applied to the dot product.
    dropout: The probability to drop units from the inputs.

  Returns:
    A tuple ``(context vector, attention vector)``.
  """
  # Dot product between queries and keys.
  dot = tf.matmul(queries, keys, transpose_b=True)

  if mask is not None:
    dot = dot * mask + ((1.0 - mask) * dot.dtype.min)

  # Compute attention weights.
  attn = tf.nn.softmax(dot)
  drop_attn = tf.layers.dropout(
      attn,
      rate=dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  # Compute attention context.
  context = tf.matmul(drop_attn, values)

  return context, attn


def multi_head_attention(num_heads,
                         queries,
                         memory,
                         mode,
                         num_units=None,
                         mask=None,
                         cache=None,
                         dropout=0.0,
                         return_attention=False):
  """Computes the multi-head attention as described in
  https://arxiv.org/abs/1706.03762.

  Args:
    num_heads: The number of attention heads.
    queries: The sequence of queries. A tensor of shape :math:`[B, T_1, ...]`.
    memory: The sequence to attend. A tensor of shape :math:`[B, T_2, ...]`.
      If ``None``, computes self-attention.
    mode: A ``tf.estimator.ModeKeys`` mode.
    num_units: The number of hidden units. If not set, it is set to the input
      dimension.
    mask: A ``tf.Tensor`` applied to the dot product.
    cache: A dictionary containing pre-projected keys and values.
    dropout: The probability to drop units from the inputs.
    return_attention: Return the attention head probabilities in addition to the
      context.

  Returns:
    The concatenated attention context of each head and the attention
    probabilities (if :obj:`return_attention` is set).
  """
  num_units = num_units or queries.get_shape().as_list()[-1]

  if num_units % num_heads != 0:
    raise ValueError("Multi head attention requires that num_units is a"
                     " multiple of {}".format(num_heads))

  if memory is None:
    queries, keys, values = fused_projection(queries, num_units, num_outputs=3)

    keys = split_heads(keys, num_heads)
    values = split_heads(values, num_heads)

    if cache is not None:
      keys = tf.concat([cache["self_keys"], keys], axis=2)
      values = tf.concat([cache["self_values"], values], axis=2)
      cache["self_keys"] = keys
      cache["self_values"] = values
  else:
    queries = tf.layers.conv1d(queries, num_units, 1)

    if cache is not None:
      def _project_and_split():
        k, v = fused_projection(memory, num_units, num_outputs=2)
        return split_heads(k, num_heads), split_heads(v, num_heads)

      keys, values = tf.cond(
          tf.equal(tf.shape(cache["memory_keys"])[2], 0),
          true_fn=_project_and_split,
          false_fn=lambda: (cache["memory_keys"], cache["memory_values"]))
      cache["memory_keys"] = keys
      cache["memory_values"] = values
    else:
      keys, values = fused_projection(memory, num_units, num_outputs=2)
      keys = split_heads(keys, num_heads)
      values = split_heads(values, num_heads)

  queries = split_heads(queries, num_heads)
  queries *= (num_units // num_heads)**-0.5

  heads, attn = dot_product_attention(
      queries,
      keys,
      values,
      mode,
      mask=mask,
      dropout=dropout)

  # Concatenate all heads output.
  combined = combine_heads(heads)
  outputs = tf.layers.conv1d(combined, num_units, 1)

  if not return_attention:
    return outputs
  return outputs, attn

def feed_forward(x, inner_dim, mode, dropout=0.0):
  """Implements the Transformer's "Feed Forward" layer.

  .. math::

      ffn(x) = max(0, x*W_1 + b_1)*W_2 + b_2

  Args:
    x: The input.
    inner_dim: The number of units of the inner linear transformation.
    mode: A ``tf.estimator.ModeKeys`` mode.
    dropout: The probability to drop units from the inner transformation.

  Returns:
    The transformed input.
  """
  input_dim = x.get_shape().as_list()[-1]

  inner = tf.layers.conv1d(x, inner_dim, 1, activation=tf.nn.relu)
  inner = tf.layers.dropout(
      inner,
      rate=dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)
  outer = tf.layers.conv1d(inner, input_dim, 1)

  return outer

def norm(inputs):
  """Layer normalizes :obj:`inputs`."""
  return tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)

def drop_and_add(inputs,
                 outputs,
                 mode,
                 dropout=0.1):
  """Drops units in the outputs and adds the previous values.

  Args:
    inputs: The input of the previous layer.
    outputs: The output of the previous layer.
    mode: A ``tf.estimator.ModeKeys`` mode.
    dropout: The probability to drop units in :obj:`outputs`.

  Returns:
    The residual and normalized output.
  """
  outputs = tf.layers.dropout(
      outputs,
      rate=dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

  input_dim = inputs.get_shape().as_list()[-1]
  output_dim = outputs.get_shape().as_list()[-1]

  if input_dim == output_dim:
    outputs += inputs
  return outputs
