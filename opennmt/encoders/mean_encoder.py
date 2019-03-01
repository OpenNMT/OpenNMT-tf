"""Define a minimal encoder."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder
from opennmt.utils import compat


class MeanEncoder(Encoder):
  """A simple encoder that takes the mean of its inputs."""

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    outputs = tf.identity(inputs)
    if sequence_length is not None and compat.tf_supports("RaggedTensor"):
      inputs = tf.RaggedTensor.from_tensor(inputs, lengths=sequence_length)
    state = tf.reduce_mean(inputs, axis=1)
    return (outputs, state, sequence_length)
