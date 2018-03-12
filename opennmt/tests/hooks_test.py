import tensorflow as tf

from opennmt.utils import hooks


class HooksTest(tf.test.TestCase):

  def testAddCounter(self):
    a = tf.placeholder(tf.int64, shape=[])
    hooks.add_counter("sum_a", a)
    sum_a = tf.get_collection(hooks._DEFAULT_COUNTERS_COLLECTION)
    self.assertIsNotNone(sum_a)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      self.assertEqual([6], sess.run(sum_a, feed_dict={a: 6}))
      self.assertEqual([10], sess.run(sum_a, feed_dict={a: 4}))


if __name__ == "__main__":
  tf.test.main()
