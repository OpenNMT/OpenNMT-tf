from parameterized import parameterized

import tensorflow as tf

from opennmt.utils import optim


class OptimTest(tf.test.TestCase):

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
    regularization = optim.regularization_penalty(
        type, scale, layer.trainable_variables)
    self.assertEqual(0, len(regularization.shape.as_list()))
    self.evaluate(regularization)

  def testRegulaizationInvalidType(self):
    with self.assertRaises(ValueError):
      optim.regularization_penalty("l3", 1e-4, [])

  def testRegulaizationMissingScaleValue(self):
    with self.assertRaises(ValueError):
      optim.regularization_penalty("l1_l2", 1e-4, [])


if __name__ == "__main__":
  tf.test.main()
