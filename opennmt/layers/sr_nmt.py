import tensorflow as tf
from opennmt.layers import transformer


def drop_and_dense_norm(inputs, num_units, num_outputs, dropout, mode):
  inputs_dp = tf.layers.dropout(
    inputs,
    rate=dropout,
    training=mode == tf.estimator.ModeKeys.TRAIN)
  output = fused_projection(
    inputs_dp,
    num_units,
    num_outputs=num_outputs
  )
  output = norm(output)
  return output


def norm(inputs):
  """Layer normalizes :obj:`inputs`."""
  return tf.contrib.layers.layer_norm(inputs, begin_norm_axis=-1)


def fused_projection(inputs, num_units, num_outputs=1):
  """Projects the same input into multiple output spaces.

  Args:
    inputs: The inputs to project.
    num_units: The number of output units of each space.
    num_outputs: The number of output spaces.

  Returns:
    :obj:`num_outputs` ``tf.Tensor`` of depth :obj:`num_units`.
  """
  return tf.layers.conv1d(inputs, num_units * num_outputs, 1, use_bias=False)


def sr_decoder_unit(inputs, state, memory, memory_mask, mode,
                    num_units, dropout, attention_dropout, num_heads):
  with tf.variable_scope("recurrent"):
    pre_act = drop_and_dense_norm(inputs,
                                  num_units,
                                  3,
                                  dropout,
                                  mode)
    pctx = drop_and_dense_norm(memory,
                               num_units,
                               1,
                               dropout,
                               mode)

    g, x, h_gate = tf.split(pre_act, num_or_size_splits=3, axis=2)
    g_sigm, h_gate_sigm = tf.sigmoid(g), tf.sigmoid(h_gate)

    def _recurrent_step(h_i, current_input):
      g_i, x_i = current_input
      h_i_new = (1. - g_i) * h_i + g_i * x_i
      return h_i_new

    # Forward step
    forward_states = tf.scan(_recurrent_step, (g_sigm, x), initializer=state)
    forward_states.set_shape(g_sigm.get_shape())
    forward_states_ = tf.transpose(forward_states, [1, 0, 2])
    new_state = forward_states_[:, -1, :]
    forward_states_dp = drop_and_dense_norm(
      forward_states_,
      num_units,
      1,
      dropout,
      mode)
  with tf.variable_scope("attention"):
    attn_out, attn = dot_product_attention(forward_states_dp,
                                     pctx,
                                     mode,
                                     memory_mask,
                                     dropout)
  with tf.variable_scope("combine"):
    trans_h = drop_and_dense_norm(
      forward_states,
      num_units,
      1,
      dropout,
      mode)
    trans_c = drop_and_dense_norm(
      attn_out,
      num_units,
      1,
      dropout,
      mode)
    out = tf.tanh(trans_h + trans_c)
  with tf.variable_scope("highway"):
    out = (1. - h_gate_sigm) * out + h_gate_sigm * inputs
  return out, new_state


def sr_encoder_unit(inputs, num_units, dropout, mode):
  # Time x batch_size x 3*dim
  pre_act = drop_and_dense_norm(inputs,
                                num_units,
                                3,
                                dropout,
                                mode)
  # 3* (Time x batch_size x dim)
  g, x, h_gate = tf.split(pre_act, num_or_size_splits=3, axis=2)
  # 2* (Time x batch_size x dim/2)
  g_f, g_b = tf.split(tf.sigmoid(g), num_or_size_splits=2, axis=2)
  x_f, x_b = tf.split(x, num_or_size_splits=2, axis=2)
  h_gate_sigm = tf.sigmoid(h_gate)
  h_f_pre = g_f * x_f
  h_b_pre = g_b * x_b

  def _recurrent_step(h_i, current_input):
    g_i, h_pre_i = current_input
    h_i_new = (1. - g_i) * h_i + h_pre_i
    return h_i_new

  # Forward step
  # batch_size x dim/2
  zero_vector = tf.zeros(tf.shape(x_f[0]))
  # Time x batch_size x dim/2
  forward_states_f = tf.scan(_recurrent_step, (g_f, h_f_pre), initializer=zero_vector)
  forward_states_f.set_shape(g_f.get_shape())  # = tf.reshape(forward_states_f, tf.shape(g_f))
  # Time x batch_size x dim/2
  forward_states_b = tf.scan(_recurrent_step,
                             (tf.reverse(g_b, axis=[0]),
                              tf.reverse(h_b_pre, axis=[0])),
                             initializer=zero_vector)
  forward_states_b.set_shape(g_b.get_shape())
  h = tf.concat([forward_states_f, tf.reverse(forward_states_b, axis=[0])], axis=2)
  output = (1. - h_gate_sigm) * h + inputs * h_gate_sigm
  return output


def dot_product_attention(queries,
                          keys,
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
  dot /= tf.sqrt(tf.cast(tf.shape(queries)[-1], dot.dtype))

  # Compute attention weights.
  attn = tf.nn.softmax(dot)
  #attn = tf.layers.dropout(
  #    attn,
  #    rate=dropout,
  #    training=mode == tf.estimator.ModeKeys.TRAIN)

  # Compute attention context.
  context = tf.matmul(attn, keys)

  return tf.transpose(context, [1, 0, 2]), attn


def multi_head_attention(num_heads,
                         queries,
                         memory,
                         mode,
                         num_units=None,
                         mask=None,
                         dropout=0.0):
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

  Returns:
    The concatenated attention context of each head.
  """
  num_units = num_units or queries.get_shape().as_list()[-1]

  if num_units % num_heads != 0:
    raise ValueError("Multi head attention requires that num_units is a"
                     " multiple of {}".format(num_heads))

  queries = tf.layers.conv1d(queries, num_units, 1)

  keys, values = transformer.fused_projection(memory, num_units, num_outputs=2)

  queries = transformer.split_heads(queries, num_heads)
  keys = transformer.split_heads(keys, num_heads)
  values = transformer.split_heads(values, num_heads)

  queries *= (num_units // num_heads)**-0.5

  heads, _ = transformer.dot_product_attention(
      queries,
      keys,
      values,
      mode,
      mask=mask,
      dropout=dropout)

  # Concatenate all heads output.
  outputs = transformer.combine_heads(heads)

  return outputs