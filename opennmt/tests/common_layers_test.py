from parameterized import parameterized

import tensorflow as tf

from opennmt.layers import common
from opennmt.tests import test_util


@test_util.run_tf2_only
class CommonLayersTest(tf.test.TestCase):

  @parameterized.expand([
      ([5, 10], [3, 5], False),
      ([10, 5], [3, 5], True),
  ])
  def testDense(self, weight_shape, input_shape, transpose):
    weight = tf.zeros(weight_shape)
    layer = common.Dense(10, weight=weight, transpose=transpose)
    x = tf.ones(input_shape)
    y = layer(x)
    self.assertEqual(layer.kernel, weight)
    self.assertEqual(self.evaluate(tf.reduce_sum(y)), 0)

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
