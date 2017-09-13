"""Define a minimal encoder."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder


class MeanEncoder(Encoder):
  """A simple encoder that takes the mean of its inputs."""

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    state = tf.reduce_mean(inputs, axis=1)
    return (inputs, state, sequence_length)
