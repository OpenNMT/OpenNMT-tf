"""Distributed optimizer."""

import tensorflow as tf
import horovod.tensorflow as hvd


class DistributedOptimizer(hvd.DistributedOptimizer):
  """This optimizer extends ``hvd.DistributedOptimizer`` to better control when
  and how gradients are reduced.
  """

  def __init__(self, optimizer, average_gradients=False, **kwargs):
    super(DistributedOptimizer, self).__init__(optimizer, **kwargs)
    self._average_gradients = average_gradients

  def compute_gradients(self, *args, **kwargs):
    """Computes the gradients by the wrapped optimizer."""
    return self._optimizer.compute_gradients(*args, **kwargs)

  def allreduce_gradients(self, grads_and_vars):
    """Reduces the gradients."""
    gradients, variables = zip(*grads_and_vars)
    with tf.name_scope(self._name + "_Allreduce"):
      if self._sparse_as_dense:
        gradients = [
            tf.convert_to_tensor(grad)
            if grad is not None and isinstance(grad, tf.IndexedSlices)
            else grad for grad in gradients]
      reduced_gradients = [
          hvd.allreduce(
              grad,
              average=self._average_gradients,
              device_dense=self._device_dense,
              device_sparse=self._device_sparse,
              compression=self._compression)
          if grad is not None else grad
          for grad in gradients]
    return list(zip(reduced_gradients, variables))

  @classmethod
  def from_params(cls, optimizer, params=None):
    """Creates an optimizer instance from user parameters.

    Args:
      optimizer: The optimizer to use.
      params: A dict of user parameters.

    Returns:
      A ``DistributedOptimizer`` instance.
    """
    if params is None:
      params = {}
    average_gradients = params.get("average_gradients", False)
    if params.get("compression", "none") == "fp16":
      compression = hvd.Compression.fp16
    else:
      compression = hvd.Compression.none
    return cls(
        optimizer,
        compression=compression,
        average_gradients=average_gradients)
