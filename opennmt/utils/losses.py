"""Define losses."""

import tensorflow as tf


def _smooth_one_hot_labels(logits, labels, label_smoothing):
  num_classes = tf.shape(logits)[-1]
  return tf.one_hot(
      tf.cast(labels, tf.int32),
      num_classes,
      on_value=1.0 - label_smoothing,
      off_value=label_smoothing / tf.to_float(num_classes - 1))

def _softmax_cross_entropy(logits, labels, label_smoothing):
  if label_smoothing > 0.0:
    smoothed_labels = _smooth_one_hot_labels(logits, labels, label_smoothing)
    return tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=smoothed_labels)
  else:
    return tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)

def cross_entropy_sequence_loss(logits,
                                labels,
                                sequence_length,
                                label_smoothing=0.0):
  """Computes the reduced cross entropy loss of sequences.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The length of each sequence.
    label_smoothing: The label smoothing value.

  Returns:
    The loss.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing)
  weights = tf.sequence_mask(sequence_length, dtype=tf.float32)
  loss = tf.reduce_sum(cross_entropy * weights) / tf.reduce_sum(weights)
  return loss

def cross_entropy_loss(logits, labels, label_smoothing=0.0):
  """Computes the reduced cross entropy loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    label_smoothing: The label smoothing value.

  Returns:
    The loss.
  """
  cross_entropy = _softmax_cross_entropy(logits, labels, label_smoothing)
  loss = tf.reduce_mean(cross_entropy)
  return loss
