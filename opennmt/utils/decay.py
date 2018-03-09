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
    learning_rate: The scale constant.
    global_step: The current learning step.
    decay_steps: The warmup steps.
    decay_rate: The model dimension.
    staircase: Ignored.
    name: Ignored.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  _ = staircase
  _ = name

  scale = tf.cast(learning_rate, tf.float32)
  step = tf.cast(global_step, tf.float32) + 1
  hidden_size = tf.cast(decay_rate, tf.float32)
  warmup_steps = tf.cast(decay_steps, tf.float32)

  return (scale
          * tf.pow(hidden_size, -0.5)
          * tf.minimum(tf.pow(step, -0.5), step * tf.pow(warmup_steps, -1.5)))


def rsqrt_decay(learning_rate,
                global_step,
                decay_steps,
                decay_rate,
                staircase=False,
                name=None):
  """Decay based on the reciprocal of the step square root.

  The semantic of the arguments are changed accordingly.

  Args:
    learning_rate: The scale constant.
    global_step: The current learning step.
    decay_steps: The warmup steps.
    decay_rate: Ignored.
    staircase: Ignored.
    name: Ignored.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  _ = decay_rate
  _ = staircase
  _ = name

  scale = tf.to_float(learning_rate)
  step = tf.to_float(global_step)
  warmup_steps = tf.to_float(decay_steps)

  return scale * tf.rsqrt(tf.maximum(step, warmup_steps))
