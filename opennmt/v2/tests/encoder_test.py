from parameterized import parameterized

import tensorflow as tf

from opennmt.v2 import encoders


class EncoderTest(tf.test.TestCase):

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testSelfAttentionEncoder(self, dtype):
    encoder = encoders.SelfAttentionEncoder(3, num_units=20, num_heads=4, ffn_inner_dim=40)
    x = tf.random.uniform([4, 5, 10], dtype=dtype)
    mask = tf.expand_dims(tf.sequence_mask([4, 3, 5, 2]), 1)
    y, _, output_mask = encoder(x, mask=mask, training=True)
    self.assertListEqual(y.shape.as_list(), [4, 5, 20])
    self.assertEqual(y.dtype, dtype)
    self.assertAllEqual(output_mask, mask)


if __name__ == "__main__":
  tf.test.main()
