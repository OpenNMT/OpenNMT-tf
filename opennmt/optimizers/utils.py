"""Optimization utilities."""

import inspect

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.optimizers.weight_decay_optimizers import DecoupledWeightDecayExtension


def make_optimizer(name, learning_rate, **kwargs):
  """Creates the optimizer.

  Args:
    name: The name of the optimizer class in ``tf.keras.optimizers`` or
      ``tfa.optimizers`` as a string.
    learning_rate: The learning rate or learning rate schedule to use.
    **kwargs: Additional optimizer arguments. If ``weight_decay`` is set, the
      optimizer will be extended with decoupled weight decay.

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
  if "weight_decay" in kwargs:
    if DecoupledWeightDecayExtension not in inspect.getmro(optimizer_class):
      optimizer_class = tfa.optimizers.extend_with_decoupled_weight_decay(optimizer_class)
  optimizer = optimizer_class(learning_rate=learning_rate, **kwargs)
  return optimizer


class GradientAccumulator(object):
  """Distribution strategies-aware gradient accumulation utility."""

  def __init__(self):
    """Initializes the accumulator."""
    self._gradients = []
    self._accum_steps = tf.Variable(
        initial_value=0,
        dtype=tf.int64,
        trainable=False,
        aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

  @property
  def step(self):
    """Number of accumulated steps."""
    return self._accum_steps.value()

  @property
  def gradients(self):
    """The accumulated gradients."""
    return list(gradient.value() for gradient in self._get_replica_gradients())

  def __call__(self, gradients):
    """Accumulates :obj:`gradients`."""
    if not self._gradients:
      self._gradients.extend([
          tf.Variable(tf.zeros_like(gradient), trainable=False)
          for gradient in gradients])
    if len(gradients) != len(self._gradients):
      raise ValueError("Expected %s gradients, but got %d" % (
          len(self._gradients), len(gradients)))

    for accum_gradient, gradient in zip(self._get_replica_gradients(), gradients):
      accum_gradient.assign_add(gradient)

    self._accum_steps.assign_add(1)

  def reset(self):
    """Resets the accumulated gradients."""
    if self._gradients:
      self._accum_steps.assign(0)
      for gradient in self._get_replica_gradients():
        gradient.assign(tf.zeros_like(gradient))

  def _get_replica_gradients(self):
    if tf.distribute.has_strategy():
      # In a replica context, we want to accumulate gradients on each replica
      # without synchronization, so we directly assign the value of the
      # current replica.
      replica_context = tf.distribute.get_replica_context()
      if replica_context is None:
        return self._gradients
      return (
          gradient.device_map.select_for_current_replica(gradient.values, replica_context)
          for gradient in self._gradients)
    else:
      return self._gradients
