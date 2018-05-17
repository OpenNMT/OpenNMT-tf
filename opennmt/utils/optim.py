"""Optimization related functions."""

import collections

import tensorflow as tf

from opennmt import optimizers
from opennmt.utils import decay


def learning_rate_decay_fn(decay_type,
                           decay_rate,
                           decay_steps,
                           decay_step_duration=1,
                           staircase=True,
                           start_decay_steps=0,
                           minimum_learning_rate=0):
  """Returns the learning rate decay functions.

  Args:
    decay_type: The type of decay. A function from ``tf.train`` or
     :mod:`opennmt.utils.decay` as a string.
    decay_rate: The decay rate to apply.
    decay_steps: The decay steps as described in the decay type function.
    decay_step_duration: The number of training steps that make 1 decay step.
    staircase: If ``True``, learning rate is decayed in a staircase fashion.
    start_decay_steps: Start decay after this many steps.
    minimum_learning_rate: Do not decay past this learning rate value.

  Returns:
    A function with signature
    ``(learning_rate, global_step) -> decayed_learning_rate``.

  Raises:
    ValueError: if :obj:`decay_type` can not be resolved.
  """
  def _decay_fn(learning_rate, global_step):
    decay_op_name = None

    if decay_op_name is None:
      decay_op_name = getattr(tf.train, decay_type, None)
    if decay_op_name is None:
      decay_op_name = getattr(decay, decay_type, None)
    if decay_op_name is None:
      raise ValueError("Unknown decay function: {}".format(decay_type))

    # Map the training step to a decay step.
    step = tf.maximum(global_step - start_decay_steps, 0)
    step = tf.div(step, decay_step_duration)

    decayed_learning_rate = decay_op_name(
        learning_rate,
        step,
        decay_steps,
        decay_rate,
        staircase=staircase)
    decayed_learning_rate = tf.maximum(decayed_learning_rate, minimum_learning_rate)

    return decayed_learning_rate

  return _decay_fn

def get_optimizer_class(classname):
  """Returns the optimizer class.

  Args:
    classname: The name of the optimizer class in ``tf.train`` or
      ``tf.contrib.opt`` as a string.

  Returns:
    A class inheriting from ``tf.train.Optimizer``.

  Raises:
    ValueError: if :obj:`classname` can not be resolved.
  """
  optimizer_class = None

  if optimizer_class is None:
    optimizer_class = getattr(tf.train, classname, None)
  if optimizer_class is None:
    optimizer_class = getattr(tf.contrib.opt, classname, None)
  if optimizer_class is None:
    optimizer_class = getattr(optimizers, classname, None)
  if optimizer_class is None:
    raise ValueError("Unknown optimizer class: {}".format(classname))

  return optimizer_class

def optimize(loss, params):
  """Minimizes the loss.

  Args:
    loss: The loss to minimize.
    params: A dictionary of hyperparameters.

  Returns:
    The loss minimization op.
  """
  global_step = tf.train.get_or_create_global_step()
  decay_type = params.get("decay_type")

  if decay_type is not None:
    decay_fn = learning_rate_decay_fn(
        decay_type,
        params["decay_rate"],
        params["decay_steps"],
        decay_step_duration=params.get("decay_step_duration", 1),
        staircase=params.get("staircase", True),
        start_decay_steps=params.get("start_decay_steps", 0),
        minimum_learning_rate=params.get("minimum_learning_rate", 0))
  else:
    decay_fn = None

  learning_rate = float(params["learning_rate"])
  clip_gradients = params.get("clip_gradients")
  if clip_gradients is not None:
    clip_gradients = float(clip_gradients)

  optimizer_class = get_optimizer_class(params["optimizer"])
  optimizer_params = params.get("optimizer_params", {})

  if optimizer_class.__name__ == "AdafactorOptimizer":
    optimizer = optimizers.get_adafactor_optimizer_from_params(optimizer_class, optimizer_params)
  else:
    optimizer = lambda lr: optimizer_class(lr, **optimizer_params)

  regularization = params.get("regularization")
  if regularization is not None:
    loss += regularization_penalty(regularization["type"], regularization["scale"])

  return tf.contrib.layers.optimize_loss(
      loss,
      global_step,
      learning_rate,
      optimizer,
      clip_gradients=clip_gradients,
      learning_rate_decay_fn=decay_fn,
      name="optim",
      summaries=[
          "learning_rate",
          "global_gradient_norm",
      ],
      colocate_gradients_with_ops=True)

def regularization_penalty(regularization_type, scale, weights_list=None):
  """Computes the weights regularization penalty.

  Args:
    regularization_type: The regularization type: ``l1``, ``l2``, or ``l1_l2``.
    scale: The regularization multiplier. If :obj:`regularization_type` is
      ``l1_l2``, this should be a list or tuple containing the L1 regularization
      scale and the L2 regularization scale.
    weights_list: The list of weights. Defaults to non bias variables.

  Returns:
    The regularization penalty.

  Raises:
    ValueError: if :obj:`regularization_type` is invalid or is ``l1_l2`` but
      :obj:`scale` is not a sequence.
  """
  def _is_bias(variable):
    return len(variable.shape.as_list()) == 1 and variable.name.endswith("bias:0")
  if weights_list is None:
    weights_list = [v for v in tf.trainable_variables() if not _is_bias(v)]

  regularization_type = regularization_type.lower()
  if regularization_type == "l1":
    regularizer = tf.contrib.layers.l1_regularizer(float(scale))
  elif regularization_type == "l2":
    regularizer = tf.contrib.layers.l2_regularizer(float(scale))
  elif regularization_type == "l1_l2":
    if not isinstance(scale, collections.Sequence) or len(scale) != 2:
      raise ValueError("l1_l2 regularization requires 2 scale values")
    regularizer = tf.contrib.layers.l1_l2_regularizer(
        scale_l1=float(scale[0]), scale_l2=float(scale[1]))
  else:
    raise ValueError("invalid regularization type %s" % regularization_type)

  return tf.contrib.layers.apply_regularization(regularizer, weights_list=weights_list)
