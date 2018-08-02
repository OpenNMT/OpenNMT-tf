"""Define reducers: objects that merge inputs."""

import abc
import six

import tensorflow as tf


def pad_in_time(x, padding_length):
  """Helper function to pad a tensor in the time dimension and retain the static depth dimension."""
  depth = x.get_shape().as_list()[-1]
  x = tf.pad(x, [[0, 0], [0, padding_length], [0, 0]])
  x.set_shape((None, None, depth))
  return x

def align_in_time(x, length):
  """Aligns the time dimension of :obj:`x` with :obj:`length`."""
  time_dim = tf.shape(x)[1]
  return tf.cond(
      tf.less(time_dim, length),
      true_fn=lambda: pad_in_time(x, length - time_dim),
      false_fn=lambda: x[:, :length])

def pad_with_identity(x, sequence_length, max_sequence_length, identity_values=0, maxlen=None):
  """Pads a tensor with identity values up to :obj:`max_sequence_length`.

  Args:
    x: A ``tf.Tensor`` of shape ``[batch_size, time, depth]``.
    sequence_length: The true sequence length of :obj:`x`.
    max_sequence_length: The sequence length up to which the tensor must contain
      :obj:`identity values`.
    identity_values: The identity value.
    maxlen: Size of the output time dimension. Default is the maximum value in
      obj:`max_sequence_length`.

  Returns:
    A ``tf.Tensor`` of shape ``[batch_size, maxlen, depth]``.
  """
  if maxlen is None:
    maxlen = tf.reduce_max(max_sequence_length)

  mask = tf.sequence_mask(sequence_length, maxlen=maxlen, dtype=x.dtype)
  mask = tf.expand_dims(mask, axis=-1)
  mask_combined = tf.sequence_mask(max_sequence_length, maxlen=maxlen, dtype=x.dtype)
  mask_combined = tf.expand_dims(mask_combined, axis=-1)

  identity_mask = mask_combined * (1.0 - mask)

  x = pad_in_time(x, maxlen - tf.shape(x)[1])
  x = x * mask + (identity_mask * identity_values)

  return x

def pad_n_with_identity(inputs, sequence_lengths, identity_values=0):
  """Pads each input tensors with identity values up to
  ``max(sequence_lengths)`` for each batch.

  Args:
    inputs: A list of ``tf.Tensor``.
    sequence_lengths: A list of sequence length.
    identity_values: The identity value.

  Returns:
    A tuple ``(padded, max_sequence_length)`` which are respectively a list of
    ``tf.Tensor`` where each tensor are padded with identity and the combined
    sequence length.
  """
  max_sequence_length = tf.reduce_max(sequence_lengths, axis=0)
  maxlen = tf.reduce_max([tf.shape(x)[1] for x in inputs])
  padded = [
      pad_with_identity(
          x, length, max_sequence_length, identity_values=identity_values, maxlen=maxlen)
      for x, length in zip(inputs, sequence_lengths)]
  return padded, max_sequence_length

def roll_sequence(tensor, offsets):
  """Shifts sequences by an offset.

  Args:
    tensor: A ``tf.Tensor`` of shape ``[batch_size, time, ...]``.
    offsets : The offset of each sequence.

  Returns:
    A ``tf.Tensor`` of the same shape as :obj:`tensor` with sequences shifted
    by :obj:`offsets`.
  """
  batch_size = tf.shape(tensor)[0]
  time = tf.shape(tensor)[1]
  cols, rows = tf.meshgrid(tf.range(time), tf.range(batch_size))
  cols -= tf.expand_dims(offsets, 1)
  cols = tf.mod(cols, time)
  indices = tf.stack([rows, cols], axis=-1)
  return tf.gather_nd(tensor, indices)


@six.add_metaclass(abc.ABCMeta)
class Reducer(object):
  """Base class for reducers."""

  def zip_and_reduce(self, x, y):
    """Zips :obj:`x` with :obj:`y` and reduces all elements."""
    if tf.contrib.framework.nest.is_sequence(x):
      tf.contrib.framework.nest.assert_same_structure(x, y)

      x_flat = tf.contrib.framework.nest.flatten(x)
      y_flat = tf.contrib.framework.nest.flatten(y)

      flat = []
      for x_i, y_i in zip(x_flat, y_flat):
        flat.append(self.reduce([x_i, y_i]))

      return tf.contrib.framework.nest.pack_sequence_as(x, flat)
    else:
      return self.reduce([x, y])

  @abc.abstractmethod
  def reduce(self, inputs):
    """Reduces all input elements.

    Args:
      inputs: A list of ``tf.Tensor``.

    Returns:
      A reduced ``tf.Tensor``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def reduce_sequence(self, inputs, sequence_lengths):
    """Reduces all input sequences.

    Args:
      inputs: A list of ``tf.Tensor``.
      sequence_lengths: The length of each input sequence.

    Returns:
      A tuple ``(reduced_input, reduced_length)`` with the reduced ``tf.Tensor``
      and sequence length.
    """
    raise NotImplementedError()


class SumReducer(Reducer):
  """A reducer that sums the inputs."""

  def reduce(self, inputs):
    return tf.add_n(inputs)

  def reduce_sequence(self, inputs, sequence_lengths):
    padded, combined_length = pad_n_with_identity(inputs, sequence_lengths, identity_values=0)
    return self.reduce(padded), combined_length


class MultiplyReducer(Reducer):
  """A reducer that multiplies the inputs."""

  def reduce(self, inputs):
    return tf.foldl(lambda a, x: a * x, tf.stack(inputs))

  def reduce_sequence(self, inputs, sequence_lengths):
    padded, combined_length = pad_n_with_identity(inputs, sequence_lengths, identity_values=1)
    return self.reduce(padded), combined_length


class ConcatReducer(Reducer):
  """A reducer that concatenates the inputs."""

  def __init__(self, axis=-1):
    self.axis = axis

  def reduce(self, inputs):
    return tf.concat(inputs, self.axis)

  def reduce_sequence(self, inputs, sequence_lengths):
    axis = self.axis % inputs[0].shape.ndims

    if axis == 2:
      padded, combined_length = pad_n_with_identity(inputs, sequence_lengths)
      return self.reduce(padded), combined_length
    elif axis == 1:
      # Align all input tensors to the maximum combined length.
      combined_length = tf.add_n(sequence_lengths)
      maxlen = tf.reduce_max(combined_length)
      aligned = [align_in_time(x, maxlen) for x in inputs]

      current_length = None
      accumulator = None

      for elem, length in zip(aligned, sequence_lengths):
        # Make sure padding are 0 vectors as it is required for the next step.
        mask = tf.sequence_mask(length, maxlen=maxlen, dtype=elem.dtype)
        elem = elem * tf.expand_dims(mask, -1)

        if accumulator is None:
          accumulator = elem
          current_length = length
        else:
          accumulator += roll_sequence(elem, current_length)
          current_length += length

      return accumulator, combined_length
    else:
      raise ValueError("Unsupported concatenation on axis {}".format(axis))


class JoinReducer(Reducer):
  """A reducer that joins its inputs in a single tuple."""

  def reduce(self, inputs):
    output = []
    for elem in inputs:
      if isinstance(elem, tuple) and not hasattr(elem, "_fields"):
        for e in elem:
          output.append(e)
      else:
        output.append(elem)
    return tuple(output)

  def reduce_sequence(self, inputs, sequence_lengths):
    return self.reduce(inputs), self.reduce(sequence_lengths)
