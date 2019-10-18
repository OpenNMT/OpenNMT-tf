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

  @parameterized.expand([
      ["ce", False],
      ["mse", False],
      ["mse", True],
  ])
  def testGuidedAlignmentCostUnderDistributionStrategy(self, cost_type, with_length):
    strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
    attention_probs = tf.random.uniform([2, 5, 6])
    gold_alignment = tf.random.uniform([2, 5, 6])
    if with_length:
      sequence_length = tf.constant([4, 5], dtype=tf.int32)
    else:
      sequence_length = None
    with strategy.scope():
      losses.guided_alignment_cost(
          attention_probs,
          gold_alignment,
          sequence_length=sequence_length,
          cost_type=cost_type)


if __name__ == "__main__":
  tf.test.main()
