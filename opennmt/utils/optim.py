"""Optimization related functions."""

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt.optimizers import lr_schedules


def make_learning_rate_schedule(initial_learning_rate,
                                schedule_type,
                                schedule_params=None,
                                schedule_step_duration=1,
                                start_step=0,
                                minimum_learning_rate=0):
  """Creates the learning rate schedule.

  Args:
    initial_learning_rate: The initial learning rate value or scale.
    schedule_type: The type of decay. A function from
      ``tf.keras.optimizers.schedules``
      or :mod:`opennmt.optimizers.lr_schedules` as a string.
    schedule_params: Additional parameters for the decay function.
    schedule_step_duration: The number of training steps that make 1 decay step.
    start_step: Start the schedule after this many steps.
    minimum_learning_rate: Do not decay past this learning rate value.

  Returns:
    A ``tf.keras.optimizers.schedules.LearningRateSchedule`` instance.

  Raises:
    ValueError: if :obj:`decay_type` can not be resolved.
  """
  schedule_name = None
  if schedule_name is None:
    schedule_name = getattr(tf.keras.optimizers.schedules, schedule_type, None)
  if schedule_name is None:
    schedule_name = getattr(lr_schedules, schedule_type, None)
  if schedule_name is None:
    raise ValueError("Unknown learning rate schedule: {}".format(schedule_type))

  if schedule_params is None:
    schedule_params = {}
  schedule = schedule_name(initial_learning_rate, **schedule_params)
  schedule = lr_schedules.ScheduleWrapper(
      schedule,
      step_start=start_step,
      step_duration=schedule_step_duration,
      minimum_learning_rate=minimum_learning_rate)
  return schedule

def make_optimizer(name, learning_rate, **kwargs):
  """Creates the optimizer.

  Args:
    name: The name of the optimizer class in ``tf.keras.optimizers`` as a string.
    learning_rate: The learning rate or learning rate schedule to use.
    **kwargs: Additional optimizer arguments.

  Returns:
    A ``tf.keras.optimizers.Optimizer`` instance.

  Raises:
    ValueError: if :obj:`name` can not be resolved to an optimizer class.
  """
  optimizer_class = None
  if optimizer_class is None:
    optimizer_class = getattr(tf.keras.optimizers, name, None)
  if optimizer_class is None:
    optimizer_class = getattr(tfa.optimizers, name, None)
  if optimizer_class is None:
    raise ValueError("Unknown optimizer class: {}".format(name))
  optimizer = optimizer_class(learning_rate=learning_rate, **kwargs)
  return optimizer

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
