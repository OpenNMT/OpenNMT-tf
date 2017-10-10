"""Define learning rate decay functions."""

import tensorflow as tf


# All functions must have the same signature.

def noam_decay(learning_rate,
               global_step,
               decay_steps,
               decay_rate,
               staircase=False,
               name=None):
  """Defines the decay function described in https://arxiv.org/abs/1706.03762.

  The semantic of the arguments are changed accordingly.

  Args:
    learning_rate: The current learning rate.
    global_step: The current learning step.
    decay_steps: The warmup steps.
    decay_rate: The model dimension.
    staircase: Ignored.
    name: Ignored.

  Returns:
    The learning rate for the step `global_step`.
  """
  _ = learning_rate
  _ = staircase
  _ = name

  global_step = tf.cast(global_step, tf.float32)
  decay_rate = tf.cast(decay_rate, tf.float32)
  decay_steps = tf.cast(decay_steps, tf.float32)

  return (tf.pow(decay_rate, -0.5)
          * tf.minimum(tf.pow(global_step, -0.5),
                       global_step * tf.pow(decay_steps, -1.5)))
