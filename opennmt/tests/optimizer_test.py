import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow_addons.optimizers.weight_decay_optimizers import (
    DecoupledWeightDecayExtension,
)

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
        self.assertIsInstance(adam_w, utils.get_optimizer_class("Adam"))
        self.assertIsInstance(adam_w, DecoupledWeightDecayExtension)

    def testCustomOptimizerRegistration(self):
        @utils.register_optimizer
        class MyCustomAdam(utils.get_optimizer_class("Adam")):
            pass

        optimizer = utils.make_optimizer("MyCustomAdam", 0.002)
        self.assertIsInstance(optimizer, MyCustomAdam)

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
                local_variables = strategy.experimental_local_results(
                    gradient_placeholder
                )
                local_variables[0].assign(grad1)
                local_variables[1].assign(grad2)
                strategy.run(accumulate_on_replica, args=(gradient_placeholder,))

        @tf.function
        def apply_grad():
            with strategy.scope():
                strategy.run(apply_on_replica)

        def _check_local_values(grad1, grad2):
            values = strategy.experimental_local_results(accumulator._gradients[0])
            self.assertAllEqual(values[0].value(), grad1)
            self.assertAllEqual(values[1].value(), grad2)

        accumulate([1.0, 2.0], [-1.0, 1.0])
        accumulate([3.0, -1.0], [-1.0, -1.0])
        accumulate([-2.0, 2.0], [3.0, -2.0])
        self.assertEqual(accumulator.step, 3)
        _check_local_values([2.0, 3.0], [1.0, -2.0])
        apply_grad()
        self.assertAllEqual(
            variable.value(), [1.0, 2.0]
        )  # [4.0 - (2.0 + 1.0), 3.0 - (3.0 - 2.0)]
        accumulator.reset()
        self.assertEqual(accumulator.step, 0)
        _check_local_values([0.0, 0.0], [0.0, 0.0])


if __name__ == "__main__":
    tf.test.main()
