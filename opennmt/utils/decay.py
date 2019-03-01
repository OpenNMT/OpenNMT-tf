"""Define learning rate decay functions."""

import tensorflow as tf
import numpy as np


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
  warmup_steps = tf.cast(warmup_steps, tf.float32)
  return scale * tf.rsqrt(tf.maximum(step, warmup_steps))


def cosine_annealing(scale, step, max_step=1000000, warmup_steps=None):
  """Decay using a cosine annealing schedule.

  Args:
    scale: The initial learning rate.
    step: The current step.
    max_step: The last step of the scedule.
    warmup_steps: The number of steps to increment the learning rate linearly
      from 0 to :obj:`scale` before annealing.

  Returns:
    The learning rate for the step :obj:`step`.
  """
  step = tf.cast(step, tf.float32)
  max_step = tf.cast(max_step, tf.float32)
  eta_min = 0
  eta_max = scale
  annealing = lambda: eta_min + 0.5 * (eta_max - eta_min) * (1 + tf.cos(np.pi * step / max_step))
  linear = lambda: scale * step / tf.cast(warmup_steps, tf.float32)
  if warmup_steps is None:
    return annealing()
  return tf.cond(tf.less(step, warmup_steps), true_fn=linear, false_fn=annealing)

def rnmtplus_decay(scale,
                   step,
                   num_replicas,
                   warmup_steps=500,
                   start_step=600000,
                   end_step=1200000):
  """Defines the decay function described in https://arxiv.org/abs/1804.09849.

  Args:
    scale: The scale constant.
    step: The current step.
    num_replicas: The number of concurrent model replicas.
    warmup_steps: The number of warmup steps.
    start_step: The start step of the exponential decay.
    end_step: The end step of the exponential decay.

  Returns:
    The learning rate for the step :obj:`step`.
  """
  t = tf.cast(step, tf.float32)
  n = tf.cast(num_replicas, tf.float32)
  p = tf.cast(warmup_steps, tf.float32)
  s = tf.cast(start_step, tf.float32)
  e = tf.cast(end_step, tf.float32)
  return scale * tf.minimum(
      tf.minimum(1 + (t * (n - 1)) / (n * p), n),
      n * tf.pow(2 * n, (s - n * t) / (e - s)))
