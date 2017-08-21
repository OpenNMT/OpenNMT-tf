import tensorflow as tf

import abc
import six

from opennmt.utils.reducer import SumReducer, JoinReducer


def create_position_embedding(embedding_size,
                              maximum_position,
                              sequence_length):
  """Creates position embeddings.

  Args:
    embedding_size: The output embedding size.
    maximum_position: The maximum position to embed.
    sequence_length: The length of each sequence of shape `[B]`.

  Returns:
    A `Tensor` of shape `[B, maximum_length, embedding_size]`.
  """
  maximum_length = tf.reduce_max(sequence_length)
  batch_size = tf.shape(sequence_length)[0]

  # Make 0 the position of padding.
  position = tf.range(maximum_length) + 1
  position = tf.tile(position, [batch_size])
  position = tf.reshape(position, [batch_size, -1])

  mask = tf.sequence_mask(sequence_length)
  mask = tf.cast(mask, tf.int32)

  position = position * mask
  position = tf.minimum(position, maximum_position)

  embeddings = tf.get_variable("w_embs", shape=[maximum_position + 1, embedding_size])
  return tf.nn.embedding_lookup(embeddings, position)


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

  def __init__(self, encoders):
    """Initializes the parameters of the encoder.

    Args:
      encoders: A list of `Encoder`.
    """
    self.encoders = encoders

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    global_state = ()

    for i in range(len(self.encoders)):
      with tf.variable_scope("encoder_" + str(i)):
        inputs, states, sequence_length = self.encoders[i].encode(
          inputs,
          sequence_length=sequence_length,
          mode=mode)
        global_state += states

    return (inputs, global_state, sequence_length)


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
