import math

import tensorflow as tf

from opennmt import decoders
from opennmt.decoders import decoder


class DecoderTest(tf.test.TestCase):

  def testSamplingProbability(self):
    step = tf.constant(5, dtype=tf.int64)
    large_step = tf.constant(1000, dtype=tf.int64)
    self.assertIsNone(decoder.get_sampling_probability(step))
    with self.assertRaises(ValueError):
      decoder.get_sampling_probability(step, schedule_type="linear")
    with self.assertRaises(ValueError):
      decoder.get_sampling_probability(step, schedule_type="linear", k=1)
    with self.assertRaises(TypeError):
      decoder.get_sampling_probability(step, schedule_type="foo", k=1)

    constant_sample_prob = decoder.get_sampling_probability(
        step, read_probability=0.9)
    linear_sample_prob = decoder.get_sampling_probability(
        step, read_probability=1.0, schedule_type="linear", k=0.1)
    linear_sample_prob_same = decoder.get_sampling_probability(
        step, read_probability=2.0, schedule_type="linear", k=0.1)
    linear_sample_prob_inf = decoder.get_sampling_probability(
        large_step, read_probability=1.0, schedule_type="linear", k=0.1)
    exp_sample_prob = decoder.get_sampling_probability(
        step, schedule_type="exponential", k=0.8)
    inv_sig_sample_prob = decoder.get_sampling_probability(
        step, schedule_type="inverse_sigmoid", k=1)

    with self.test_session() as sess:
      self.assertAlmostEqual(0.1, constant_sample_prob)
      self.assertAlmostEqual(0.5, sess.run(linear_sample_prob))
      self.assertAlmostEqual(0.5, sess.run(linear_sample_prob_same))
      self.assertAlmostEqual(1.0, sess.run(linear_sample_prob_inf))
      self.assertAlmostEqual(1.0 - pow(0.8, 5), sess.run(exp_sample_prob))
      self.assertAlmostEqual(
          1.0 - (1.0 / (1.0 + math.exp(5.0 / 1.0))), sess.run(inv_sig_sample_prob))


if __name__ == "__main__":
  tf.test.main()
