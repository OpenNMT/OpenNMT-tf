from parameterized import parameterized

import tensorflow as tf

from opennmt.v2 import encoders


class EncoderTest(tf.test.TestCase):

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testSelfAttentionEncoder(self, dtype):
    encoder = encoders.SelfAttentionEncoder(3, num_units=20, num_heads=4, ffn_inner_dim=40)
    inputs = tf.random.uniform([4, 5, 10], dtype=dtype)
    lengths = tf.constant([4, 3, 5, 2])
    outputs, _, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(outputs.dtype, dtype)

  @parameterized.expand([[tf.keras.layers.LSTMCell], [tf.keras.layers.GRUCell]])
  def testUnidirectionalRNNEncoder(self, cell_class):
    encoder = encoders.UnidirectionalRNNEncoder(3, 20, cell_class=cell_class)
    inputs = tf.random.uniform([4, 5, 10])
    lengths = tf.constant([4, 3, 5, 2])
    outputs, states, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(len(states), 3)

  @parameterized.expand([[tf.keras.layers.LSTMCell], [tf.keras.layers.GRUCell]])
  def testBidirectionalRNNEncoder(self, cell_class):
    encoder = encoders.BidirectionalRNNEncoder(3, 20, cell_class=cell_class)
    inputs = tf.random.uniform([4, 5, 10])
    lengths = tf.constant([4, 3, 5, 2])
    outputs, states, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(len(states), 3)


if __name__ == "__main__":
  tf.test.main()
