"""Utilities to run execution in parallel."""

import six

import tensorflow as tf

from tensorflow.python.client import device_lib
from tensorflow.python.estimator.util import fn_args


class GraphDispatcher(object):
  """Helper class to replicate graph parts on multiple devices and dispatch
  sharded batches.
  """

  def __init__(self,
               num_devices=None,
               daisy_chain_variables=True,
               devices=None,
               session_config=None):
    """Initializes the dispatcher.

    Args:
      num_devices: The number of devices to dispatch on.
      daisy_chain_variables: If ``True``, variables are copied in a daisy chain
        fashion between devices (credits to Tensor2Tensor).
      devices: List of devices to use (takes priority over :obj:`num_devices`).
      session_config: Session configuration to use when querying available
        devices.

    Raises:
      ValueError: if the number of visible devices is lower than
        :obj:`num_devices`.
    """
    if devices:
      self._devices = devices
    elif num_devices is not None:
      self._devices = get_devices(num_devices=num_devices, session_config=session_config)
    else:
      self._devices = [None]
    self._n = len(self._devices)
    self._daisy_chain_variables = daisy_chain_variables

  def shard(self, data):
    """Shards a structure of ``tf.Tensor`` for dispatching.

    Args:
      data: A ``tf.Tensor`` of dictionary of ``tf.Tensor``.

    Returns:
      A list of the same ``tf.Tensor`` structure.
    """
    return split_batch(data, self._n)

  def repeat(self, data):
    """Ensures that the object is dispatchable list.

    Args:
      data: The object to convert.

    Returns:
      :obj:`data` if it is valid list or a list where :obj:`data` is replicated.

    Raises:
      ValueError: if :obj:`data` is a non dispatchable list.
    """
    if isinstance(data, list):
      if len(data) != self._n:
        raise ValueError("List arguments must contain %d elements, saw %d instead"
                         % (self._n, len(data)))
      return data
    else:
      return [data] * self._n

  def _parallel_args(self, *args, **kwargs):
    """Makes each argument dispatchable."""
    if args:
      parallel_args = [self.repeat(arg) for arg in args]
      parallel_args = [list(arg) for arg in zip(*parallel_args)]
    else:
      parallel_args = [[] for _ in range(self._n)]
    parallel_kwargs = [{} for _ in range(self._n)]
    for k, v in six.iteritems(kwargs):
      values = self.repeat(v)
      for i in range(self._n):
        parallel_kwargs[i][k] = values[i]
    return parallel_args, parallel_kwargs

  def __call__(self, fun, *args, **kwargs):
    """Dispatches :obj:`fun` calls accross devices.

    Each argument must either not be a list or a list with length the number of
    devices used for dispatching.

    Args:
      fun: A callable.
      *args: The callable arguments.
      **kwargs: The callable keyword arguments.

    Returns:
      The sharded outputs of :obj:`fun`.
    """
    funs = self.repeat(fun)
    args, kwargs = self._parallel_args(*args, **kwargs)

    outputs = []
    cache = {}
    tensor_to_var = {}

    for i, device in enumerate(self._devices):

      # pylint: disable=cell-var-from-loop
      def _daisy_chain_getter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        # Copyright 2017 The Tensor2Tensor Authors.
        # Licensed under the Apache License, Version 2.0
        device_var_key = (device, name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          last_device_v = cache[name]
          var = tensor_to_var[last_device_v]
          v = tf.identity(last_device_v)
        else:
          var = getter(name, *args, **kwargs)
          v = tf.identity(var._ref())  # pylint: disable=protected-access

        # keep track of the original variable
        tensor_to_var[v] = var
        v.read_value = lambda: tf.identity(v)
        v.assign_sub = var.assign_sub
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      custom_getter = None
      if self._daisy_chain_variables:
        custom_getter = _daisy_chain_getter

      with tf.variable_scope(
          tf.get_variable_scope(),
          reuse=True if i > 0 else None,
          custom_getter=custom_getter):
        if device is None:
          outputs.append(funs[i](*args[i], **kwargs[i]))
        else:
          with tf.device(device):
            outputs.append(funs[i](*args[i], **kwargs[i]))

    # If the function returned a tuple, also return a tuple of sharded results.
    if isinstance(outputs[0], tuple):
      outputs = tuple(list(output) for output in zip(*outputs))

    return outputs


def split_batch(data, num_shards):
  """Split data into shards."""

  def _split_dictionary(dictionary):
    """Split a dictionary into shards."""
    shards = [{} for _ in range(num_shards)]
    for name, tensor in six.iteritems(dictionary):
      if isinstance(tensor, tf.SparseTensor):
        for i, shard in enumerate(tf.sparse_split(sp_input=tensor, num_split=num_shards, axis=0)):
          shards[i][name] = shard
      else:
        for i, shard in enumerate(tf.split(tensor, num_shards)):
          shards[i][name] = shard
    return shards

  with tf.name_scope("split_inputs"):
    if data is None:
      data_shards = None
    elif isinstance(data, dict):
      data_shards = _split_dictionary(data)
    else:
      data_shards = tf.split(data, num_shards)

  return data_shards

def get_devices(num_devices=None, session_config=None):
  """Returns available devices.

  Args:
    num_devices: The number of devices to get.
    session_config: An optional session configuration to use when querying
      available devices.

  Returns:
    A list of devices.

  Raises:
    ValueError: if :obj:`num_devices` is set but the number of visible devices
      is lower than it.
  """
  kwargs = {}
  if "session_config" in fn_args(device_lib.list_local_devices):
    kwargs["session_config"] = session_config
  else:
    # Create a first session to enforce config, otherwise list_local_devices()
    # will run some initialization with default options.
    _ = tf.Session(config=session_config)
  devices = [x.name for x in device_lib.list_local_devices(**kwargs) if x.device_type == "GPU"]
  if not devices:
    return [None]
  elif num_devices is None:
    return devices
  elif len(devices) < num_devices:
    raise ValueError("Only %d devices are visible but %d were requested"
                     % (len(devices), num_devices))
  return devices[:num_devices]
