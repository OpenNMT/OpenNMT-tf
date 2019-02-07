import tensorflow as tf

from opennmt.v2.layers import common


class CommonLayersTest(tf.test.TestCase):

  def testLayerNorm(self):
    layer_norm = common.LayerNorm()
    x = tf.random.uniform([4, 10])
    y = layer_norm(x)
    self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
  tf.test.main()
