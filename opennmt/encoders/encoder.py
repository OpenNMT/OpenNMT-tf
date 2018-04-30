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
  """An encoder that executes multiple encoders sequentially with optional
  transition layers.

  See for example "Cascaded Encoder" in https://arxiv.org/abs/1804.09849.
  """

  def __init__(self, encoders, states_reducer=JoinReducer(), transition_layer_fn=None):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of :class:`opennmt.encoders.encoder.Encoder`.
      states_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        states.
      transition_layer_fn: A callable or list of callables applied to the
        output of an encoder before passing it as input to the next. If it is a
        single callable, it is applied between every encoders. Otherwise, the
        ``i`` th callable will be applied between encoders ``i`` and ``i + 1``.

    Raises:
      ValueError: if :obj:`transition_layer_fn` is a list with a size not equal
        to the number of encoder transitions ``len(encoders) - 1``.
    """
    if (transition_layer_fn is not None and isinstance(transition_layer_fn, list)
        and len(transition_layer_fn) != len(encoders) - 1):
      raise ValueError("The number of transition layers must match the number of encoder "
                       "transitions, expected %d layers but got %d."
                       % (len(encoders) - 1, len(transition_layer_fn)))
    self.encoders = encoders
    self.states_reducer = states_reducer
    self.transition_layer_fn = transition_layer_fn

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    encoder_state = []

    for i, encoder in enumerate(self.encoders):
      with tf.variable_scope("encoder_{}".format(i)):
        if i > 0 and self.transition_layer_fn is not None:
          if isinstance(self.transition_layer_fn, list):
            inputs = self.transition_layer_fn[i - 1](inputs)
          else:
            inputs = self.transition_layer_fn(inputs)
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
  """An encoder that encodes its input with several encoders and reduces the
  outputs and states together. Additional layers can be applied on each encoder
  output and on the combined output (e.g. to layer normalize each encoder
  output).

  If the input is a single ``tf.Tensor``, the same input will be encoded by
  every encoders. Otherwise, when the input is a Python sequence (e.g. the non
  reduced output of a :class:`opennmt.inputters.inputter.ParallelInputter`),
  each encoder will encode its corresponding input in the sequence.

  See for example "Multi-Columnn Encoder" in https://arxiv.org/abs/1804.09849.
  """

  def __init__(self,
               encoders,
               outputs_reducer=ConcatReducer(axis=1),
               states_reducer=JoinReducer(),
               outputs_layer_fn=None,
               combined_output_layer_fn=None):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of :class:`opennmt.encoders.encoder.Encoder`.
      outputs_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        outputs.
      states_reducer: A :class:`opennmt.layers.reducer.Reducer` to merge all
        states.
      outputs_layer_fn: A callable or list of callables applied to the
        encoders outputs If it is a single callable, it is on each encoder
        output. Otherwise, the ``i`` th callable is applied on encoder ``i``
        output.
      combined_output_layer_fn: A callable to apply on the combined output
        (i.e. the output of :obj:`outputs_reducer`).

    Raises:
      ValueError: if :obj:`outputs_layer_fn` is a list with a size not equal
        to the number of encoders.
    """
    if (outputs_layer_fn is not None and isinstance(outputs_layer_fn, list)
        and len(outputs_layer_fn) != len(encoders)):
      raise ValueError("The number of output layers must match the number of encoders; "
                       "expected %d layers but got %d."
                       % (len(encoders), len(outputs_layer_fn)))
    self.encoders = encoders
    self.outputs_reducer = outputs_reducer
    self.states_reducer = states_reducer
    self.outputs_layer_fn = outputs_layer_fn
    self.combined_output_layer_fn = combined_output_layer_fn

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

        if self.outputs_layer_fn is not None:
          if isinstance(self.outputs_layer_fn, list):
            outputs = self.outputs_layer_fn[i](outputs)
          else:
            outputs = self.outputs_layer_fn(outputs)

        all_outputs.append(outputs)
        all_states.append(state)
        all_sequence_lengths.append(length)

    outputs, sequence_length = self.outputs_reducer.reduce_sequence(
        all_outputs, all_sequence_lengths)

    if self.combined_output_layer_fn is not None:
      outputs = self.combined_output_layer_fn(outputs)

    return (outputs, self.states_reducer.reduce(all_states), sequence_length)
