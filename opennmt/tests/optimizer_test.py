import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.optimizers.weight_decay_optimizers import DecoupledWeightDecayExtension

from opennmt.optimizers import utils


class OpimizerTest(tf.test.TestCase):

  def testMakeLazyAdam(self):
    lazy_adam = utils.make_optimizer("LazyAdam", 0.002, beta_1=0.8)
    self.assertIsInstance(lazy_adam, tfa.optimizers.LazyAdam)
    self.assertEqual(lazy_adam.learning_rate, 0.002)
    self.assertEqual(lazy_adam.beta_1, 0.8)

  def testMakeAdamW(self):
    adam_w = utils.make_optimizer("AdamW", 0.002, weight_decay=0.1)
    self.assertIsInstance(adam_w, tfa.optimizers.AdamW)
    adam_w = utils.make_optimizer("Adam", 0.002, weight_decay=0.1)
    self.assertIsInstance(adam_w, tf.keras.optimizers.Adam)
    self.assertIsInstance(adam_w, DecoupledWeightDecayExtension)


if __name__ == "__main__":
  tf.test.main()
