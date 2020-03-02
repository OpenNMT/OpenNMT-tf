import tensorflow as tf

from opennmt import training
from opennmt.tests import test_util


class TrainingTest(tf.test.TestCase):

  def testMovingAverage(self):
    step = tf.Variable(0, trainable=False, dtype=tf.int64)
    variables = [tf.Variable(1.0), tf.Variable(2.0)]
    moving_average = training.MovingAverage(variables, step)
    variables[0].assign(3.0)
    variables[1].assign(4.0)
    moving_average.update()
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])
    with moving_average.shadow_variables():
      self.assertAllClose(self.evaluate(variables), [2.8, 3.8])
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])

  @test_util.run_with_two_cpu_devices
  def testMovingAverageDistributionStrategy(self):
    devices = tf.config.experimental.list_logical_devices(device_type="CPU")
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    with strategy.scope():
      variables = [tf.Variable(1.0), tf.Variable(2.0)]
      step = tf.Variable(0, trainable=False, dtype=tf.int64)

    moving_average = training.MovingAverage(variables, step)
    variables[0].assign(3.0)
    variables[1].assign(4.0)
    moving_average.update()
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])
    with moving_average.shadow_variables():
      self.assertAllClose(self.evaluate(variables), [2.8, 3.8])
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])


if __name__ == "__main__":
  tf.test.main()
