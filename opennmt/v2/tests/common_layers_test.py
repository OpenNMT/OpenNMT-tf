import tensorflow as tf

from opennmt.v2.layers import common


class CommonLayersTest(tf.test.TestCase):

  def testLayerNorm(self):
    layer_norm = common.LayerNorm()
    x = tf.random.uniform([4, 10])
    y = layer_norm(x)
    self.assertEqual(y.shape, x.shape)

  def testLayerWrapper(self):
    layer = common.LayerWrapper(tf.keras.layers.Dense(10))
    x = tf.random.uniform([4, 5, 10])
    y = layer(x)
    self.assertEqual(y.shape, x.shape)

  def testLayerWrapperInputOutputDepthMismatch(self):
    layer = common.LayerWrapper(tf.keras.layers.Dense(10))
    x = tf.random.uniform([4, 5, 5])
    y = layer(x)
    self.assertListEqual(y.shape.as_list(), [4, 5, 10])


if __name__ == "__main__":
  tf.test.main()
