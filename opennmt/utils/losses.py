"""Define losses."""

import tensorflow as tf


def _smooth_one_hot_labels(logits, labels, label_smoothing):
  label_smoothing = tf.constant(label_smoothing, dtype=logits.dtype)
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.cast(num_classes - 1, label_smoothing.dtype),
      dtype=logits.dtype)

def _softmax_cross_entropy(logits, labels, label_smoothing, training):
  # Computes the softmax in full precision.
  if logits.dtype.base_dtype != tf.float32:
    logits = tf.cast(logits, tf.float32)
  if training and label_smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    return tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                label_smoothing=0.0,
                                average_in_time=False,
                                training=None):
  """Computes the cross entropy loss of sequences.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The length of each sequence.
    label_smoothing: The label smoothing value.
    average_in_time: If ``True``, also average the loss in the time dimension.
    training: Compute training loss. If not set, infer training mode from
      :obj:`mode`.

  Returns:
    A tuple (cumulated loss, loss normalizer, token-level normalizer).
  """
  batch_size = tf.shape(logits)[0]
  max_time = tf.shape(logits)[1]

  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)
  weights = tf.sequence_mask(
      sequence_length, maxlen=max_time, dtype=cross_entropy.dtype)
  loss = tf.reduce_sum(cross_entropy * weights)
  loss_token_normalizer = tf.reduce_sum(weights)

  if average_in_time or not training:
    loss_normalizer = loss_token_normalizer
  else:
    loss_normalizer = tf.cast(batch_size, loss.dtype)

  return loss, loss_normalizer, loss_token_normalizer

def cross_entropy_loss(logits,
                       labels,
                       label_smoothing=0.0,
                       training=None):
  """Computes the cross entropy loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    label_smoothing: The label smoothing value.
    training: Compute training loss. If not set, infer training mode from
      :obj:`mode`.

  Returns:
    The cumulated loss and the loss normalizer.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)
  loss = tf.reduce_sum(cross_entropy)
  loss_normalizer = tf.cast(tf.shape(cross_entropy)[0], loss.dtype)
  return loss, loss_normalizer

def guided_alignment_cost(attention_probs,
                          gold_alignment,
                          sequence_length=None,
                          cost_type="ce",
                          weight=1):
  """Computes the guided alignment cost.

  Args:
    attention_probs: The attention probabilities, a float ``tf.Tensor`` of shape
      :math:`[B, T_t, T_s]`.
    gold_alignment: The true alignment matrix, a float ``tf.Tensor`` of shape
      :math:`[B, T_t, T_s]`.
    sequence_length: The length of each sequence.
    cost_type: The type of the cost function to compute (can be: ce, mse).
    weight: The weight applied to the cost.

  Returns:
    The guided alignment cost.

  Raises:
    ValueError: if :obj:`cost_type` is invalid.
  """
  if cost_type == "ce":
    loss = tf.keras.losses.CategoricalCrossentropy()
  elif cost_type == "mse":
    loss = tf.keras.losses.MeanSquaredError()
  else:
    raise ValueError("invalid guided alignment cost: %s" % cost_type)

  if sequence_length is not None:
    sample_weight = tf.sequence_mask(
        sequence_length,
        maxlen=tf.shape(attention_probs)[1],
        dtype=attention_probs.dtype)
    sample_weight = tf.expand_dims(sample_weight, -1)
  else:
    sample_weight = None

  cost = loss(
      gold_alignment,
      attention_probs,
      sample_weight=sample_weight)
  return weight * cost

def regularization_penalty(regularization_type, scale, weights):
  """Computes the weights regularization penalty.

  Args:
    regularization_type: The regularization type: ``l1``, ``l2``, or ``l1_l2``.
    scale: The regularization multiplier. If :obj:`regularization_type` is
      ``l1_l2``, this should be a list or tuple containing the L1 regularization
      scale and the L2 regularization scale.
    weights: The list of weights.

  Returns:
    The regularization penalty.

  Raises:
    ValueError: if :obj:`regularization_type` is invalid or is ``l1_l2`` but
      :obj:`scale` is not a sequence.
  """
  regularization_type = regularization_type.lower()
  if regularization_type == "l1":
    regularizer = tf.keras.regularizers.l1(l=float(scale))
  elif regularization_type == "l2":
    regularizer = tf.keras.regularizers.l2(l=float(scale))
  elif regularization_type == "l1_l2":
    if not isinstance(scale, (list, tuple)) or len(scale) != 2:
      raise ValueError("l1_l2 regularization requires 2 scale values")
    regularizer = tf.keras.regularizers.l1_l2(
        l1=float(scale[0]), l2=float(scale[1]))
  else:
    raise ValueError("invalid regularization type %s" % regularization_type)

  weights = list(filter(lambda v: not _is_bias(v), weights))
  penalty = tf.add_n([regularizer(w) for w in weights])
  return penalty

def _is_bias(variable):
  return len(variable.shape.as_list()) == 1 and variable.name.endswith("bias:0")
