"""Base class for encoders and generic multi encoders."""

import abc
import six

import tensorflow as tf

from opennmt.layers.reducer import ConcatReducer, JoinReducer


@six.add_metaclass(abc.ABCMeta)
class Encoder(object):
  """Base class for encoders."""

  @abc.abstractmethod
  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    """Encodes an input sequence.

    Args:
      inputs: The inputs to encode of shape :math:`[B, T, ...]`.
      sequence_length: The length of each input with shape :math:`[B]`.
      mode: A ``tf.estimator.ModeKeys`` mode.

    Returns:
      A tuple ``(outputs, state, sequence_length)``.
    """
    raise NotImplementedError()


class SequentialEncoder(Encoder):
  """An encoder that executes multiple encoders sequentially."""

  def __init__(self, encoders, states_reducer=JoinReducer()):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of :class:`opennmt.encoders.encoder.Encoder`.
      states_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        states.
    """
    self.encoders = encoders
    self.states_reducer = states_reducer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_state = []

    for i, encoder in enumerate(self.encoders):
      with tf.variable_scope("encoder_{}".format(i)):
        inputs, state, sequence_length = encoder.encode(
            inputs,
            sequence_length=sequence_length,
            mode=mode)
        encoder_state.append(state)

    return (
        inputs,
        self.states_reducer.reduce(encoder_state),
        sequence_length)


class ParallelEncoder(Encoder):
  """An encoder that encodes inputs with several encoders. If the input
  is a sequence, each encoder will encode its corresponding input in the
  sequence. Otherwise, the same input will be encoded by every encoders.
  """

  def __init__(self,
               encoders,
               outputs_reducer=ConcatReducer(axis=1),
               states_reducer=JoinReducer()):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of :class:`opennmt.encoders.encoder.Encoder`.
      outputs_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        outputs.
      states_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        states.
    """
    self.encoders = encoders
    self.outputs_reducer = outputs_reducer
    self.states_reducer = states_reducer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    all_outputs = []
    all_states = []
    all_sequence_lengths = []

    if tf.contrib.framework.nest.is_sequence(inputs) and len(inputs) != len(self.encoders):
      raise ValueError("ParallelEncoder expects as many inputs as parallel encoders")

    for i, encoder in enumerate(self.encoders):
      with tf.variable_scope("encoder_{}".format(i)):
        if tf.contrib.framework.nest.is_sequence(inputs):
          encoder_inputs = inputs[i]
          length = sequence_length[i]
        else:
          encoder_inputs = inputs
          length = sequence_length

        outputs, state, length = encoder.encode(
            encoder_inputs,
            sequence_length=length,
            mode=mode)

        all_outputs.append(outputs)
        all_states.append(state)
        all_sequence_lengths.append(length)

    outputs, sequence_length = self.outputs_reducer.reduce_sequence(
        all_outputs, all_sequence_lengths)

    return (outputs, self.states_reducer.reduce(all_states), sequence_length)
