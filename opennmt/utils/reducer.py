"""Define reducers: objects that merge inputs."""

import abc
import six

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Reducer(object):
  """Base class for reducers."""

  @abc.abstractmethod
  def reduce_all(self, inputs):
    """Reduces all tensors in the list `inputs`."""
    raise NotImplementedError()

  def reduce(self, x, y):
    """Reduce `x` and `y`."""
    if tf.contrib.framework.nest.is_sequence(x):
      tf.contrib.framework.nest.assert_same_structure(x, y)

      x_flat = tf.contrib.framework.nest.flatten(x)
      y_flat = tf.contrib.framework.nest.flatten(y)

      flat = []
      for x_i, y_i in zip(x_flat, y_flat):
        flat.append(self.reduce_all([x_i, y_i]))

      return tf.contrib.framework.nest.pack_sequence_as(x, flat)
    else:
      return self.reduce_all([x, y])


class SumReducer(Reducer):
  """A reducer that sums the inputs."""

  def reduce_all(self, inputs):
    return tf.add_n(inputs)


class MultiplyReducer(Reducer):
  """A reducer that multiplies the inputs."""

  def reduce_all(self, inputs):
    return tf.foldl(lambda a, x: a * x, inputs)


class ConcatReducer(Reducer):
  """A reducer that concatenates the inputs alognside the last axis."""

  def reduce_all(self, inputs):
    return tf.concat(inputs, -1)


class JoinReducer(Reducer):
  """A reducer that joins its inputs in a single tuple."""

  def reduce_all(self, inputs):
    output = ()
    for elem in inputs:
      if isinstance(elem, tuple):
        output += elem
      else:
        output += (elem,)
    return output
