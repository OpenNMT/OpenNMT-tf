import tensorflow as tf

import abc
import six

from opennmt.utils.reducer import SumReducer, JoinReducer


@six.add_metaclass(abc.ABCMeta)
class Encoder(object):
  """Abstract class for encoders."""

  @abc.abstractmethod
  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    """Encodes an input sequence.

    Args:
      inputs: The input to encode of shape [B, T, ...].
      sequence_length: The length of each input with shape [B].
      mode: A `tf.estimator.ModeKeys` mode.

    Returns:
      A tuple (`encoder_outputs`, `encoder_states`, `encoder_sequence_length`).
    """
    raise NotImplementedError()


class SequentialEncoder(Encoder):
  """An encoder that executes multiple encoders sequentially."""

  def __init__(self, encoders, states_reducer=JoinReducer()):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of `Encoder`.
      states_reducer: A `Reducer` to merge all states.
    """
    self.encoders = encoders
    self.states_reducer = states_reducer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_state = []

    for i in range(len(self.encoders)):
      with tf.variable_scope("encoder_" + str(i)):
        inputs, state, sequence_length = self.encoders[i].encode(
          inputs,
          sequence_length=sequence_length,
          mode=mode)
        encoder_state.append(state)

    encoder_state = self.states_reducer.reduce_all(encoder_state)

    return (inputs, encoder_state, sequence_length)


class ParallelEncoder(Encoder):
  """An encoder that encodes inputs with several encoders. If the input
  is a sequence, each encoder will encode its corresponding input in the
  sequence. Otherwise, the same input will be encoded by every encoders.
  """

  def __init__(self,
               encoders,
               outputs_reducer=SumReducer(),
               states_reducer=JoinReducer()):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of `Encoder`.
      outputs_reducer: A `Reducer` to merge all outputs.
      states_reducer: A `Reducer` to merge all states.
    """
    self.encoders = encoders
    self.outputs_reducer = outputs_reducer
    self.states_reducer = states_reducer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    all_outputs = []
    all_states = []

    if tf.contrib.framework.nest.is_sequence(inputs) and len(inputs) != len(self.encoders):
      raise ValueError("ParallelEncoder expects as many inputs as parallel encoders")

    # TODO: execute in parallel?
    for i in range(len(self.encoders)):
      with tf.variable_scope("encoder_" + str(i)):
        if tf.contrib.framework.nest.is_sequence(inputs):
          encoder_inputs = inputs[i]
        else:
          encoder_inputs = inputs

        outputs, states, sequence_length = self.encoders[i].encode(
          encoder_inputs,
          sequence_length=sequence_length,
          mode=mode)
        all_outputs.append(outputs)
        all_states.append(states)

    return (
      self.outputs_reducer.reduce_all(all_outputs),
      self.states_reducer.reduce_all(all_states),
      sequence_length)
