"""Define a minimal encoder."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder


class MeanEncoder(Encoder):
    """A simple encoder that takes the mean of its inputs."""

    def call(self, inputs, sequence_length=None, training=None):
        outputs = tf.identity(inputs)
        if sequence_length is not None:
            inputs = tf.RaggedTensor.from_tensor(inputs, lengths=sequence_length)
        state = tf.reduce_mean(inputs, axis=1)
        return (outputs, state, sequence_length)
