"""Define losses."""

import tensorflow as tf


def _smooth_one_hot_labels(logits, labels, label_smoothing):
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.to_float(num_classes - 1))

def _softmax_cross_entropy(logits, labels, label_smoothing, mode):
  if mode == tf.estimator.ModeKeys.TRAIN and label_smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=smoothed_labels)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                label_smoothing=0.0,
                                average_in_time=False,
                                mode=tf.estimator.ModeKeys.TRAIN):
  """Computes the reduced cross entropy loss of sequences.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The length of each sequence.
    label_smoothing: The label smoothing value.
    average_in_time: If ``True``, also average the loss in the time dimension.
    mode: A ``tf.estimator.ModeKeys`` mode.

  Returns:
    The loss.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, mode)
  weights = tf.sequence_mask(sequence_length, dtype=tf.float32)
  loss = tf.reduce_sum(cross_entropy * weights)
  normalized_loss = loss / tf.reduce_sum(weights)

  # Summarize the normalized loss for better interpretability.
  tf.summary.scalar("normalized_loss", normalized_loss)

  if average_in_time:
    return normalized_loss
  else:
    batch_size = tf.shape(logits)[0]
    return loss / tf.to_float(batch_size)

def cross_entropy_loss(logits,
                       labels,
                       label_smoothing=0.0,
                       mode=tf.estimator.ModeKeys.TRAIN):
  """Computes the reduced cross entropy loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    label_smoothing: The label smoothing value.
    mode: A ``tf.estimator.ModeKeys`` mode.

  Returns:
    The loss.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing, mode)
  loss = tf.reduce_mean(cross_entropy)
  return loss
