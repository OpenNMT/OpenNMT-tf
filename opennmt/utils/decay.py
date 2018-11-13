"""Define learning rate decay functions."""

import tensorflow as tf


# All functions must take the learning rate and the step as first arguments.

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
  return noam_decay_v2(learning_rate, global_step, decay_rate, decay_steps)

def noam_decay_v2(scale, step, model_dim, warmup_steps):
  """Defines the decay function described in https://arxiv.org/abs/1706.03762.

  Args:
    scale: The scale constant.
    step: The current step.
    model_dim: The model dimension.
    warmup_steps: The number of warmup steps.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  step = tf.cast(step + 1, tf.float32)
  model_dim = tf.cast(model_dim, tf.float32)
  warmup_steps = tf.cast(warmup_steps, tf.float32)
  return (scale
          * tf.pow(model_dim, -0.5)
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
  return rsqrt_decay_v2(learning_rate, global_step, decay_steps)

def rsqrt_decay_v2(scale, step, warmup_steps):
  """Decay based on the reciprocal of the step square root.

  Args:
    scale: The scale constant.
    step: The current step.
    warmup_steps: The number of warmup steps.

  Returns:
    The learning rate for the step :obj:`global_step`.
  """
  step = tf.cast(step, tf.float32)
  warmup_steps = tf.cast(warmum_steps, tf.float32)
  return scale * tf.rsqrt(tf.maximum(step, warmup_steps))
