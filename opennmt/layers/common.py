"""Defines common layers."""

import tensorflow as tf

from tensorflow.python.framework import function


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
  """Wraps :obj:`x` to convert its gradient to a tensor."""
  return x


def embedding_lookup(params, ids):
  """Wrapper around ``tf.nn.embedding_lookup``.

  This converts gradients of the embedding variable to tensors which allows
  to use of optimizers that don't support sparse gradients (e.g. Adafactor).

  Args:
    params: The embedding tensor.
    ids: The ids to lookup in :obj:`params`.

  Returns:
    A ``tf.Tensor``, the embeddings that correspond to :obj:`ids`.
  """
  params = convert_gradient_to_tensor(params)
  return tf.nn.embedding_lookup(params, ids)
