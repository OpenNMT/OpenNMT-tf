"""Define bridges: logic of passing the encoder state to the decoder."""

import abc
import six

import tensorflow as tf

from opennmt.utils import compat


def assert_state_is_compatible(expected_state, state):
  """Asserts that states are compatible.

  Args:
    expected_state: The reference state.
    state: The state that must be compatible with :obj:`expected_state`.

  Raises:
    ValueError: if the states are incompatible.
  """
  # Check structure compatibility.
  compat.nest.assert_same_structure(expected_state, state)

  # Check shape compatibility.
  expected_state_flat = compat.nest.flatten(expected_state)
  state_flat = compat.nest.flatten(state)

  for x, y in zip(expected_state_flat, state_flat):
    if compat.is_tensor(x):
      expected_depth = x.get_shape().as_list()[-1]
      depth = y.get_shape().as_list()[-1]
      if depth != expected_depth:
        raise ValueError("Tensor in state has shape %s which is incompatible "
                         "with the target shape %s" % (y.shape, x.shape))


@six.add_metaclass(abc.ABCMeta)
class Bridge(tf.keras.layers.Layer):
  """Base class for bridges."""

  def __call__(self, encoder_state, decoder_zero_state):  # pylint: disable=arguments-differ
    """Returns the initial decoder state.

    Args:
      encoder_state: The encoder state.
      decoder_zero_state: The default decoder state.

    Returns:
      The decoder initial state.
    """
    inputs = [encoder_state, decoder_zero_state]
    if compat.is_tf2():
      return super(Bridge, self).__call__(inputs)
    # Build by default for backward compatibility.
    if not compat.reuse():
      self.build(compat.nest.map_structure(lambda x: x.shape, inputs))
    return self.call(inputs)

  @abc.abstractmethod
  def call(self, states):  # pylint: disable=arguments-differ
    raise NotImplementedError()


class CopyBridge(Bridge):
  """A bridge that passes the encoder state as is."""

  def call(self, states):
    assert_state_is_compatible(states[0], states[1])
    return states[0]


class ZeroBridge(Bridge):
  """A bridge that does not pass information from the encoder."""

  def call(self, states):
    # Simply return the default decoder state.
    return states[1]


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
    super(DenseBridge, self).__init__()
    self.activation = activation
    self.decoder_state_sizes = None
    self.linear = None

  def build(self, input_shape):
    decoder_shape = input_shape[1]
    self.decoder_state_sizes = [
        shape.as_list()[-1] for shape in compat.nest.flatten(decoder_shape)]
    self.linear = tf.keras.layers.Dense(
        sum(self.decoder_state_sizes),
        activation=self.activation,
        name=compat.name_from_variable_scope("dense"))

  def call(self, states):
    encoder_state, decoder_state = states
    encoder_state_flat = compat.nest.flatten(encoder_state)
    encoder_state_single = tf.concat(encoder_state_flat, 1)
    transformed = self.linear(encoder_state_single)
    splitted = tf.split(transformed, self.decoder_state_sizes, axis=1)
    return compat.nest.pack_sequence_as(decoder_state, splitted)
