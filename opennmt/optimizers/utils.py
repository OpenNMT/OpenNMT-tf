"""Optimization utilities."""

import inspect

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.optimizers.weight_decay_optimizers import DecoupledWeightDecayExtension


def get_optimizer_class(name):
  """Returns the optimizer class.

  Args:
    name: The optimizer name.

  Returns:
    A class extending ``tf.keras.optimizers.Optimizer``.

  Raises:
    ValueError: if :obj:`name` can not be resolved to an optimizer class.
  """
  optimizer_class = None
  if optimizer_class is None:
    optimizer_class = getattr(tf.keras.optimizers, name, None)
  if optimizer_class is None:
    optimizer_class = getattr(tfa.optimizers, name, None)
  if optimizer_class is None:
    raise ValueError("Unknown optimizer class: %s" % name)
  return optimizer_class

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
  optimizer_class = get_optimizer_class(name)
  if "weight_decay" in kwargs:
    if DecoupledWeightDecayExtension not in inspect.getmro(optimizer_class):
      optimizer_class = tfa.optimizers.extend_with_decoupled_weight_decay(optimizer_class)
  optimizer = optimizer_class(learning_rate=learning_rate, **kwargs)
  return optimizer


class GradientAccumulator(object):
  """Gradient accumulation utility.

  When used with a distribution strategy, the accumulator should be called in a
  replica context. Gradients will be accumulated locally on each replica and
  without synchronization. Users should then call ``.gradients``, scale the
  gradients if required, and pass the result to ``apply_gradients``.
  """

  def __init__(self):
    """Initializes the accumulator."""
    self._gradients = []
    self._accum_steps = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)

  @property
  def step(self):
    """Number of accumulated steps."""
    return self._accum_steps.value()

  @property
  def gradients(self):
    """The accumulated gradients on the current replica."""
    if not self._gradients:
      raise ValueError("The accumulator should be called first to initialize the gradients")
    return list(gradient.value() for gradient in _get_replica_local_variables(self._gradients))

  def __call__(self, gradients):
    """Accumulates :obj:`gradients` on the current replica."""
    if not self._gradients:
      self._gradients.extend([
          tf.Variable(tf.zeros_like(gradient), trainable=False)
          for gradient in gradients])
    if len(gradients) != len(self._gradients):
      raise ValueError("Expected %s gradients, but got %d" % (
          len(self._gradients), len(gradients)))

    # In a replica context, we want to accumulate gradients on each replica
    # without synchronization, so we directly assign the value of the
    # current replica.
    for accum_gradient, gradient in zip(_get_replica_local_variables(self._gradients), gradients):
      accum_gradient.assign_add(gradient)
    _get_replica_local_variables(self._accum_steps).assign_add(1)

  def reset(self):
    """Resets the accumulated gradients on the current replica."""
    if self._gradients:
      _get_replica_local_variables(self._accum_steps).assign(0)
      for gradient in _get_replica_local_variables(self._gradients):
        gradient.assign(tf.zeros_like(gradient))


def _get_replica_local_variables(variables):
  if not tf.distribute.has_strategy():
    return variables
  if tf.distribute.in_cross_replica_context():
    raise RuntimeError("GradientAccumulator should be used in a replica context")
  replica_context = tf.distribute.get_replica_context()
  replica_id = replica_context.replica_id_in_sync_group
  if not isinstance(replica_id, int):
    replica_id = tf.get_static_value(replica_id)
  return tf.nest.map_structure(lambda variable: variable.values[replica_id], variables)
