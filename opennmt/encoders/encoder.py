import tensorflow as tf

import abc
import six


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

    for encoder in self.encoders:
      inputs, states, sequence_length = encoder.encode(
        inputs,
        sequence_length=sequence_length,
        mode=mode)
      global_state += states

    return (inputs, global_state, sequence_length)
