import tensorflow as tf
from opennmt.encoders.encoder import Encoder
from opennmt.layers import sr_nmt as srnmt

class LNMASRUBiEncoder(Encoder):

  def __init__(self,
               num_layers,
               num_units,
               dropout=0.1):
    self.num_layers = num_layers
    self.num_units = num_units
    self.dropout = dropout

  def encode(self, inputs, sequence_length=None, mode=tf.estimator.ModeKeys.TRAIN):
    # Time x batch_size x dim
    inputs = tf.transpose(inputs, [1, 0, 2])
    zero_vector = tf.zeros(tf.shape(inputs[0]))

    for l in range(self.num_layers):
      with tf.variable_scope("layer_{}".format(l)):
        layer_out = srnmt.sr_encoder_unit(inputs, self.num_units, self.dropout, mode)
        inputs = layer_out

    output = tf.transpose(layer_out, [1, 0, 2])
    return output, zero_vector, sequence_length


