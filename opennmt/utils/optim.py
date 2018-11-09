"""Optimization related functions."""

import collections

import tensorflow as tf

from opennmt import optimizers
from opennmt.utils import decay
from opennmt.optimizers.mixed_precision_wrapper import get_loss_scale_from_params


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

def optimize(*args, **kwargs):
  """Wrapper around ``optimize_loss`` for backward compatibility."""
  update_op, _ = optimize_loss(*args, **kwargs)
  return update_op

def optimize_loss(loss, params, mixed_precision=False):
  """Minimizes the loss.

  Args:
    loss: The loss to minimize.
    params: A dictionary of hyperparameters.
    mixed_precision: If ``True``, wraps the optimizer to maintain a float32 copy
      of the weights.

  Returns:
    The loss minimization op and a list of internal variables to initialize.
  """
  regularization = params.get("regularization")
  if regularization is not None:
    loss += regularization_penalty(regularization["type"], regularization["scale"])

  global_step = tf.train.get_or_create_global_step()
  with tf.variable_scope("optim"):
    # Learning rate.
    learning_rate = tf.get_variable(
        "learning_rate",
        [],
        trainable=False,
        initializer=tf.constant_initializer(float(params["learning_rate"])))
    if "decay_type" in params:
      decay_fn = learning_rate_decay_fn(
          params["decay_type"],
          params["decay_rate"],
          params["decay_steps"],
          decay_step_duration=params.get("decay_step_duration", 1),
          staircase=params.get("staircase", True),
          start_decay_steps=params.get("start_decay_steps", 0),
          minimum_learning_rate=params.get("minimum_learning_rate", 0))
      learning_rate = decay_fn(learning_rate, global_step)
    tf.summary.scalar("learning_rate", learning_rate)

    # Optimizer.
    optimizer_class = get_optimizer_class(params["optimizer"])
    optimizer_params = params.get("optimizer_params", {})
    if optimizer_class.__name__ == "AdafactorOptimizer":
      optimizer = optimizers.get_adafactor_optimizer_from_params(
          optimizer_class, optimizer_params, learning_rate=learning_rate)
    else:
      optimizer = optimizer_class(learning_rate, **optimizer_params)
    if mixed_precision:
      optimizer = optimizers.MixedPrecisionOptimizerWrapper(
          optimizer, loss_scale=get_loss_scale_from_params(params))

    # Gradients.
    gradients = optimizer.compute_gradients(loss, colocate_gradients_with_ops=True)
    _summarize_gradients_norm("global_norm/gradient_norm", gradients)
    if "clip_gradients" in params:
      gradients = _clip_gradients_by_norm(gradients, float(params["clip_gradients"]))
      _summarize_gradients_norm("global_norm/clipped_gradient_norm", gradients)

    return delayed_update(
        optimizer,
        gradients,
        global_step,
        accum_count=params.get("gradients_accum_steps", 1))

def delayed_update(optimizer, grads_and_vars, global_step, accum_count=1):
  """Possibly delays the parameters update by first accumulating gradients.

  Args:
    optimizer: The optimizer.
    grads_and_vars: List of (gradient, variable) pairs.
    global_step: The current training step.
    accum_count: The number of steps to accumulate gradients.

  Returns:
    An operation that conditionally applies the gradients and a list of internal
    variables to initialize.
  """
  if accum_count == 1:
    return optimizer.apply_gradients(grads_and_vars, global_step=global_step), []

  accum_grads = []
  accum_grads_and_vars = []
  for grad, var in grads_and_vars:
    accum_grad = tf.Variable(tf.zeros_like(grad), trainable=False, collections=[])
    accum_grads.append(accum_grad)
    accum_grads_and_vars.append((accum_grad, var))

  def _accum_grads(accum_fn=tf.assign_add, apply_gradients=False):
    update_ops = []
    for accum_grad, (grad, _) in zip(accum_grads, grads_and_vars):
      with tf.control_dependencies([grad]):
        update_ops.append(accum_fn(accum_grad, grad))
    with tf.control_dependencies(update_ops):
      if apply_gradients:
        return optimizer.apply_gradients(accum_grads_and_vars, global_step=global_step)
      else:
        with tf.control_dependencies([global_step.assign_add(1)]):
          return tf.no_op()

  update_op = tf.cond(
      tf.equal((global_step + 1) % accum_count, 0),
      true_fn=lambda: _accum_grads(apply_gradients=True),
      false_fn=lambda: tf.cond(
          tf.equal(global_step % accum_count, 0),
          true_fn=lambda: _accum_grads(accum_fn=tf.assign),
          false_fn=_accum_grads))
  return update_op, accum_grads

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

def _clip_gradients_by_norm(grads_and_vars, clip_gradients):
  """Clips gradients by global norm."""
  gradients, variables = zip(*grads_and_vars)
  clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_gradients)
  return list(zip(clipped_gradients, variables))

def _summarize_gradients_norm(name, gradients):
  """Summarizes global norm of gradients."""
  tf.summary.scalar(name, tf.global_norm(list(zip(*gradients))[0]))
