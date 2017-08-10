"""Define convolution-based encoders."""

import tensorflow as tf

from opennmt.encoders.encoder import Encoder, create_position_embedding
from opennmt.utils.reducer import SumReducer


class ConvEncoder(Encoder):
  """An encoder that applies a convolution over the input sequence
  as described in https://arxiv.org/abs/1611.02344.
  """

  def __init__(self,
               num_layers,
               num_units,
               kernel_size=3,
               dropout=0.3,
               position_embedding=True,
               position_embedding_max=128,
               position_embedding_reducer=SumReducer()):
    """Initializes the parameters of the encoder.

    Args:
      num_layers: The number of convolutional layers.
      num_units: The number of output filters.
      kernel_size: The kernel size.
      dropout: The probability to drop units from the inputs.
      position_embedding: If `True`, add position embedding.
      position_embedding_max: Maximum position.
      position_embedding_reducer: A `Reducer` to merge inputs and position
        embeddings.
    """
    self.num_layers = num_layers
    self.num_units = num_units
    self.kernel_size = kernel_size
    self.dropout = dropout

    self.position_embedding = position_embedding
    self.position_embedding_max = position_embedding_max
    self.position_embedding_reducer = position_embedding_reducer

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    if self.position_embedding:
      with tf.variable_scope("position_embedding"):
        input_dim = inputs.get_shape().as_list()[-1]
        position_embedding = create_position_embedding(
          input_dim,
          self.position_embedding_max,
          sequence_length)
        inputs = self.position_embedding_reducer.reduce(inputs, position_embedding)

    # Apply dropout to inputs.
    inputs = tf.layers.dropout(
      inputs,
      rate=self.dropout,
      training=mode == tf.estimator.ModeKeys.TRAIN)

    cnn_a = self._cnn_stack(inputs, "cnn_a")
    cnn_c = self._cnn_stack(inputs, "cnn_c")

    encoder_output = cnn_a
    encoder_state = tf.reduce_mean(cnn_c, axis=1)

    return (encoder_output, encoder_state, sequence_length)

  def _cnn_stack(self, inputs, scope_name):
    with tf.variable_scope(scope_name):
      next_input = inputs

      for l in range(self.num_layers):
        outputs = tf.contrib.layers.conv2d(
          inputs=next_input,
          num_outputs=self.num_units,
          kernel_size=self.kernel_size,
          padding="SAME",
          activation_fn=None)

        # Add residual connections past the first layer.
        if l > 0:
          outputs += next_input

        next_input = tf.tanh(outputs)

      return next_input
