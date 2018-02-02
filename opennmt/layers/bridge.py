"""Define bridges: logic of passing the encoder state to the decoder."""

import abc
import six

import tensorflow as tf


def assert_state_is_compatible(expected_state, state):
  """Asserts that states are compatible.

  Args:
    expected_state: The reference state.
    state: The state that must be compatible with :obj:`expected_state`.

  Raises:
    ValueError: if the states are incompatible.
  """
  # Check structure compatibility.
  tf.contrib.framework.nest.assert_same_structure(expected_state, state)

  # Check shape compatibility.
  expected_state_flat = tf.contrib.framework.nest.flatten(expected_state)
  state_flat = tf.contrib.framework.nest.flatten(state)

  for x, y in zip(expected_state_flat, state_flat):
    if tf.contrib.framework.is_tensor(x):
      tf.contrib.framework.with_same_shape(x, y)


@six.add_metaclass(abc.ABCMeta)
class Bridge(object):
  """Base class for bridges."""

  def __call__(self, encoder_state, decoder_zero_state):
    """Returns the initial decoder state.

    Args:
      encoder_state: The encoder state.
      decoder_zero_state: The default decoder state.

    Returns:
      The decoder initial state.
    """
    return self._build(encoder_state, decoder_zero_state)

  @abc.abstractmethod
  def _build(self, encoder_state, decoder_zero_state):
    raise NotImplementedError()


class CopyBridge(Bridge):
  """A bridge that passes the encoder state as is."""

  def _build(self, encoder_state, decoder_zero_state):
    assert_state_is_compatible(decoder_zero_state, encoder_state)
    return encoder_state


class ZeroBridge(Bridge):
  """A bridge that does not pass information from the encoder."""

  def _build(self, encoder_state, decoder_zero_state):
    # Simply return the default decoder state.
    return decoder_zero_state


class DenseBridge(Bridge):
  """A bridge that applies a parameterized linear transformation from the
  encoder state to the decoder state size.
  """

  def __init__(self, activation=None):
    """Initializes the bridge.

    Args:
      activation: Activation function (a callable).
        Set it to ``None`` to maintain a linear activation.
    """
    self.activation = activation

  def _build(self, encoder_state, decoder_zero_state):
    # Flattened states.
    encoder_state_flat = tf.contrib.framework.nest.flatten(encoder_state)
    decoder_state_flat = tf.contrib.framework.nest.flatten(decoder_zero_state)

    # View encoder state as a single tensor.
    encoder_state_concat = tf.concat(encoder_state_flat, 1)

    # Extract decoder state sizes.
    decoder_state_size = []
    for tensor in decoder_state_flat:
      decoder_state_size.append(tensor.get_shape().as_list()[-1])

    decoder_total_size = sum(decoder_state_size)

    # Apply linear transformation.
    transformed = tf.layers.dense(
        encoder_state_concat,
        decoder_total_size,
        activation=self.activation)

    # Split resulting tensor to match the decoder state size.
    splitted = tf.split(transformed, decoder_state_size, axis=1)

    # Pack as the origial decoder state.
    return tf.contrib.framework.nest.pack_sequence_as(decoder_zero_state, splitted)
