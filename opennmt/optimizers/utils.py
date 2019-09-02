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
