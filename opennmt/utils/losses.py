"""Define losses."""

import tensorflow as tf


def masked_sequence_loss(logits, labels, sequence_length):
  """Builds a masked sequence loss.

  Args:
    logits: The unscaled probabilities.
    labels: The true labels.
    sequence_length: The size of each sequence.

  Returns:
    The loss.
  """
  mask = tf.sequence_mask(sequence_length)
  weights = tf.cast(mask, tf.float32)

  return tf.contrib.seq2seq.sequence_loss(
    logits,
    labels,
    weights)
