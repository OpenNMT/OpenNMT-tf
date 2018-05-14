import tensorflow as tf
import numpy as np

from opennmt.utils import optim


class OptimTest(tf.test.TestCase):

  def _testRegularization(self, type, scale):
    tf.reset_default_graph()
    x = tf.placeholder_with_default(
        np.random.randn(64, 128).astype(np.float32), shape=(None, 128))
    x = tf.layers.dense(x, 256)
    x = tf.layers.dense(x, 128)
    regularization = optim.regularization_penalty(type, scale)
    self.assertEqual(0, len(regularization.shape.as_list()))
    with self.test_session(tf.get_default_graph()) as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(regularization)

  def testRegularization(self):
    self._testRegularization("l1", 1e-4)
    self._testRegularization("L1", 1e-4)
    self._testRegularization("l1", 1)
    self._testRegularization("l2", 1e-4)
    self._testRegularization("l1_l2", (1e-4, 1e-4))
    with self.assertRaises(ValueError):
      self._testRegularization("l1_l2", 1e-4)
    with self.assertRaises(ValueError):
      self._testRegularization("l3", 1e-4)


if __name__ == "__main__":
  tf.test.main()
