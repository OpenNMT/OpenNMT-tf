from parameterized import parameterized

import tensorflow as tf

from opennmt.utils import losses


class LossesTest(tf.test.TestCase):

  @parameterized.expand([
      ["l1", 1e-4],
      ["L1", 1e-4],
      ["l1", 1],
      ["l2", 1e-4],
      ["l1_l2", (1e-4, 1e-4)],
  ])
  def testRegularization(self, type, scale):
    layer = tf.keras.layers.Dense(256)
    layer.build([None, 128])
    regularization = losses.regularization_penalty(
        type, scale, layer.trainable_variables)
    self.assertEqual(0, len(regularization.shape))
    self.evaluate(regularization)

  def testRegulaizationInvalidType(self):
    with self.assertRaises(ValueError):
      losses.regularization_penalty("l3", 1e-4, [])

  def testRegulaizationMissingScaleValue(self):
    with self.assertRaises(ValueError):
      losses.regularization_penalty("l1_l2", 1e-4, [])


if __name__ == "__main__":
  tf.test.main()
