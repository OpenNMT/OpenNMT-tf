"""Define losses."""

import tensorflow as tf


def _smooth_one_hot_labels(logits, labels, label_smoothing):
    num_classes = logits.shape[-1]
    on_value = 1.0 - label_smoothing
    off_value = label_smoothing / (num_classes - 1)
    return tf.one_hot(
        labels,
        num_classes,
        on_value=tf.cast(on_value, logits.dtype),
        off_value=tf.cast(off_value, logits.dtype),
    )


def _softmax_cross_entropy(logits, labels, label_smoothing, training):
    # Computes the softmax in full precision.
    logits = tf.cast(logits, tf.float32)
    if training and label_smoothing > 0.0:
        smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
        return tf.nn.softmax_cross_entropy_with_logits(smoothed_labels, logits)
    else:
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)


def cross_entropy_sequence_loss(
    logits,
    labels,
    sequence_length=None,
    label_smoothing=0.0,
    average_in_time=False,
    training=None,
    sequence_weight=None,
):
    """Computes the cross entropy loss of sequences.

    Args:
      logits: The unscaled probabilities with shape :math:`[B, T, V]`.
      labels: The true labels with shape :math:`[B, T]`.
      sequence_length: The length of each sequence with shape :math:`[B]`.
      label_smoothing: The label smoothing value.
      average_in_time: If ``True``, also average the loss in the time dimension.
      training: Compute training loss.
      sequence_weight: The weight of each sequence with shape :math:`[B]`.

    Returns:
      A tuple (cumulated loss, loss normalizer, token-level normalizer).
    """
    cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)
    dtype = cross_entropy.dtype

    shape = tf.shape(logits)
    batch_size = shape[0]
    max_time = shape[1]

    if sequence_length is None:
        sequence_length = tf.fill([batch_size], max_time)

    weight = tf.sequence_mask(sequence_length, maxlen=max_time, dtype=dtype)
    if sequence_weight is not None:
        sequence_weight = tf.cast(sequence_weight, dtype)
        weight *= tf.expand_dims(sequence_weight, 1)

    loss = tf.reduce_sum(cross_entropy * weight)

    loss_token_normalizer = tf.reduce_sum(weight)
    if average_in_time or not training:
        loss_normalizer = loss_token_normalizer
    elif sequence_weight is not None:
        loss_normalizer = tf.reduce_sum(sequence_weight)
    else:
        loss_normalizer = tf.cast(batch_size, dtype)

    return loss, loss_normalizer, loss_token_normalizer


def cross_entropy_loss(logits, labels, label_smoothing=0.0, training=None, weight=None):
    """Computes the cross entropy loss.

    Args:
      logits: The unscaled probabilities with shape :math:`[B, V]`.
      labels: The true labels with shape :math:`[B]`.
      label_smoothing: The label smoothing value.
      training: Compute training loss.
      weight: The weight of each example with shape :math:`[B]`.

    Returns:
      The cumulated loss and the loss normalizer.
    """
    cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, training)

    if weight is not None:
        weight = tf.cast(weight, cross_entropy.dtype)
        cross_entropy *= weight
        loss_normalizer = tf.reduce_sum(weight)
    else:
        batch_size = tf.shape(cross_entropy)[0]
        loss_normalizer = tf.cast(batch_size, cross_entropy.dtype)

    loss = tf.reduce_sum(cross_entropy)
    return loss, loss_normalizer


def guided_alignment_cost(
    attention_probs, gold_alignment, sequence_length=None, cost_type="ce", weight=1
):
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
        loss = tf.keras.losses.CategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM
        )
    elif cost_type == "mse":
        loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    else:
        raise ValueError("invalid guided alignment cost: %s" % cost_type)

    if sequence_length is not None:
        sample_weight = tf.sequence_mask(
            sequence_length,
            maxlen=tf.shape(attention_probs)[1],
            dtype=attention_probs.dtype,
        )
        sample_weight = tf.expand_dims(sample_weight, -1)
        normalizer = tf.reduce_sum(sequence_length)
    else:
        sample_weight = None
        normalizer = tf.size(attention_probs)

    cost = loss(gold_alignment, attention_probs, sample_weight=sample_weight)
    cost /= tf.cast(normalizer, cost.dtype)
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
            l1=float(scale[0]), l2=float(scale[1])
        )
    else:
        raise ValueError("invalid regularization type %s" % regularization_type)

    weights = list(filter(lambda v: not _is_bias(v), weights))
    penalty = tf.add_n([regularizer(w) for w in weights])
    return penalty


def _is_bias(variable):
    return len(variable.shape) == 1 and variable.name.endswith("bias:0")


def _negative_log_likelihood(logits, labels, sequence_length):
    nll_num, nll_den, _ = cross_entropy_sequence_loss(
        logits, labels, sequence_length, average_in_time=True
    )
    return nll_num / nll_den


def max_margin_loss(
    true_logits,
    true_labels,
    true_sequence_length,
    negative_logits,
    negative_labels,
    negative_sequence_length,
    eta=0.1,
):
    """Computes the max-margin loss described in
    https://www.aclweb.org/anthology/P19-1623.

    Args:
      true_logits: The unscaled probabilities from the true example.
      negative_logits: The unscaled probabilities from the negative example.
      true_labels: The true labels.
      true_sequence_length: The length of each true sequence.
      negative_labels: The negative labels.
      negative_sequence_length: The length of each negative sequence.
      eta: Ensure that the margin is higher than this value.

    Returns:
      The max-margin loss.
    """
    true_nll = _negative_log_likelihood(true_logits, true_labels, true_sequence_length)
    negative_nll = _negative_log_likelihood(
        negative_logits, negative_labels, negative_sequence_length
    )
    margin = true_nll - negative_nll + eta
    return tf.maximum(margin, 0)
