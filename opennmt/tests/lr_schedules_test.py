import tensorflow as tf

from parameterized import parameterized

from opennmt.schedules import lr_schedules


class _IdentitySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __call__(self, step):
        return step


class LRSchedulesTest(tf.test.TestCase):
    def _testSchedule(self, schedule, expected_values):
        for i, expected_value in enumerate(expected_values):
            step = tf.constant(i, dtype=tf.int64)
            value = schedule(step)
            self.assertEqual(self.evaluate(value), expected_value)

    def _testNoError(self, schedule):
        step = tf.constant(1, dtype=tf.int64)
        schedule(step)

    def testGetScheduleClass(self):
        with self.assertRaises(ValueError):
            lr_schedules.get_lr_schedule_class("ScheduleWrapper")
        lr_schedules.get_lr_schedule_class("NoamDecay") == lr_schedules.NoamDecay

    def testMakeSchedule(self):
        wrapper = lr_schedules.make_learning_rate_schedule(
            2.0, "ExponentialDecay", dict(decay_steps=1000, decay_rate=0.7)
        )
        self.assertIsInstance(
            wrapper.schedule, tf.keras.optimizers.schedules.ExponentialDecay
        )

        wrapper = lr_schedules.make_learning_rate_schedule(
            2.0, "NoamDecay", dict(model_dim=512, warmup_steps=4000)
        )
        self.assertIsInstance(wrapper.schedule, lr_schedules.NoamDecay)
        self.assertEqual(wrapper.schedule.scale, 2)

        wrapper = lr_schedules.make_learning_rate_schedule(
            None, "NoamDecay", dict(scale=2, model_dim=512, warmup_steps=4000)
        )
        self.assertEqual(wrapper.schedule.scale, 2)

        with self.assertRaises(ValueError):
            lr_schedules.make_learning_rate_schedule(2.0, "InvalidScheduleName")

    def testScheduleWrapper(self):
        self._testSchedule(
            lr_schedules.ScheduleWrapper(_IdentitySchedule()), [0, 1, 2, 3, 4]
        )
        self._testSchedule(
            lr_schedules.ScheduleWrapper(_IdentitySchedule(), step_start=2),
            [0, 0, 0, 1, 2, 3],
        )
        self._testSchedule(
            lr_schedules.ScheduleWrapper(_IdentitySchedule(), step_duration=2),
            [0, 0, 1, 1, 2, 2],
        )
        self._testSchedule(
            lr_schedules.ScheduleWrapper(_IdentitySchedule(), minimum_learning_rate=2),
            [2, 2, 2, 3, 4, 5],
        )

    def testNoamDecay(self):
        self._testNoError(lr_schedules.NoamDecay(2.0, 512, 4000))

    def testRsqrtDecay(self):
        self._testNoError(lr_schedules.RsqrtDecay(2.0, 4000))

    @parameterized.expand([(0,), (1e-07,)])
    def testInvSqrtDecay(self, initial_learning_rate):
        learning_rate = 0.0002
        warmup_steps = 4000
        schedule = lr_schedules.InvSqrtDecay(
            learning_rate, warmup_steps, initial_learning_rate=initial_learning_rate
        )
        self.assertNotEqual(schedule(0), initial_learning_rate)
        self.assertEqual(schedule(warmup_steps - 1), learning_rate)

    def testCosineAnnealing(self):
        self._testNoError(
            lr_schedules.CosineAnnealing(2.5e-4, max_step=1000000, warmup_steps=4000)
        )

    def testRNMTPlusDecay(self):
        self._testNoError(lr_schedules.RNMTPlusDecay(1.0, 2))


if __name__ == "__main__":
    tf.test.main()
