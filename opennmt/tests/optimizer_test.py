import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.optimizers.weight_decay_optimizers import DecoupledWeightDecayExtension

from opennmt.optimizers import utils
from opennmt.tests import test_util


class OptimizerTest(tf.test.TestCase):

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

  def testGradientAccumulator(self):
    accumulator = utils.GradientAccumulator()
    accumulator([tf.constant([1.0, 2.0])])
    accumulator([tf.constant([-2.0, 1.0])])
    accumulator([tf.constant([-1.0, 2.0])])
    with self.assertRaises(ValueError):
      accumulator([tf.constant([1.0, 1.0]), tf.constant([2.0, 2.0])])
    self.assertEqual(accumulator.step, 3)
    self.assertEqual(len(accumulator.gradients), 1)
    self.assertAllEqual(accumulator.gradients[0], [-2.0, 5.0])
    accumulator.reset()
    self.assertEqual(accumulator.step, 0)
    self.assertAllEqual(accumulator.gradients[0], [0.0, 0.0])

  @test_util.run_with_two_cpu_devices
  def testGradientAccumulatorDistributionStrategy(self):
    devices = tf.config.list_logical_devices(device_type="CPU")
    strategy = tf.distribute.MirroredStrategy(devices=devices[:2])

    with strategy.scope():
      accumulator = utils.GradientAccumulator()
      variable = tf.Variable([4.0, 3.0])
      sgd = tf.keras.optimizers.SGD(1.0)
      gradient_placeholder = tf.Variable([0.0, 0.0], trainable=False)

    def accumulate_on_replica(gradient):
      accumulator([gradient])

    def apply_on_replica():
      sgd.apply_gradients(list(zip(accumulator.gradients, [variable])))

    @tf.function
    def accumulate(grad1, grad2):
      with strategy.scope():
        gradient_placeholder.values[0].assign(grad1)
        gradient_placeholder.values[1].assign(grad2)
        strategy.experimental_run_v2(accumulate_on_replica, args=(gradient_placeholder,))

    @tf.function
    def apply_grad():
      with strategy.scope():
        strategy.experimental_run_v2(apply_on_replica)

    accumulate([1.0, 2.0], [-1.0, 1.0])
    accumulate([3.0, -1.0], [-1.0, -1.0])
    accumulate([-2.0, 2.0], [3.0, -2.0])
    self.assertEqual(accumulator.step, 3)
    self.assertAllEqual(accumulator._gradients[0].values[0].value(), [2.0, 3.0])
    self.assertAllEqual(accumulator._gradients[0].values[1].value(), [1.0, -2.0])
    apply_grad()
    self.assertAllEqual(variable.value(), [1.0, 2.0])  # [4.0 - (2.0 + 1.0), 3.0 - (3.0 - 2.0)]
    accumulator.reset()
    self.assertEqual(accumulator.step, 0)
    self.assertAllEqual(accumulator._gradients[0].values[0].value(), [0.0, 0.0])
    self.assertAllEqual(accumulator._gradients[0].values[1].value(), [0.0, 0.0])


if __name__ == "__main__":
  tf.test.main()
