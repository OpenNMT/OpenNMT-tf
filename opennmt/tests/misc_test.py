import tensorflow as tf

from opennmt.utils import misc


class MiscTest(tf.test.TestCase):

  def _testLinearSchedule(self,
                          expected_value,
                          global_step,
                          initial_value,
                          target_value=None,
                          delay_steps=None,
                          gain_steps=None):
    global_step = tf.constant(global_step, dtype=tf.int64)
    value = misc.linear_schedule(
        global_step,
        initial_value,
        target_value=target_value,
        delay_steps=delay_steps,
        gain_steps=gain_steps)
    self.assertIsInstance(value, tf.Tensor)
    self.assertEqual(value.dtype, tf.float32)
    value_fp16 = misc.linear_schedule(
        global_step,
        initial_value,
        target_value=target_value,
        delay_steps=delay_steps,
        gain_steps=gain_steps,
        dtype=tf.float16)
    self.assertIsInstance(value_fp16, tf.Tensor)
    self.assertEqual(value_fp16.dtype, tf.float16)
    with self.test_session() as sess:
      self.assertNear(expected_value, sess.run(value), 1e-8)
      self.assertNear(expected_value, sess.run(value_fp16), 1e-4)

  def testLinearScheduleConstant(self):
    self._testLinearSchedule(0.1, 0, 0.1)
    self._testLinearSchedule(0.1, 100, 0.1)
    self._testLinearSchedule(0.0, 100, 0.0)

  def testLinearScheduleStairCase(self):
    self._testLinearSchedule(0.0, 1, 0.1, target_value=0.0)
    self._testLinearSchedule(0.1, 60, 0.1, target_value=0.0, delay_steps=100)
    self._testLinearSchedule(0.1, 100, 0.1, target_value=0.0, delay_steps=100)
    self._testLinearSchedule(0.0, 101, 0.1, target_value=0.0, delay_steps=100)
    self._testLinearSchedule(0.1, 101, 0.0, target_value=0.1, delay_steps=100)

  def testLinearScheduleLinear(self):
    self._testLinearSchedule(0.1, 50, 0.1,
                             target_value=0.0, delay_steps=100, gain_steps=100)
    self._testLinearSchedule(0.09, 110, 0.1,
                             target_value=0.0, delay_steps=100, gain_steps=100)
    self._testLinearSchedule(0.01, 110, 0.0,
                             target_value=0.1, delay_steps=100, gain_steps=100)
    self._testLinearSchedule(0.1, 300, 0.0,
                             target_value=0.1, delay_steps=100, gain_steps=100)


if __name__ == "__main__":
  tf.test.main()
