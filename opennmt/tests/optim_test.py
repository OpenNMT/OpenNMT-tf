import tensorflow as tf
import numpy as np

from opennmt.utils import optim
from opennmt.tests import test_util


@test_util.run_tf1_only
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

  def testDelayedUpdate(self):
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    gradient = tf.placeholder(tf.float32, shape=[2])
    variable = tf.Variable([1.0, 2.0])
    train_op, extra_variables = optim.delayed_update(
        optimizer,
        [(gradient, variable)],
        global_step,
        accum_count=3)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(tf.variables_initializer(extra_variables))

      def _check_step(grad, expected_variable, expected_step):
        _, variable_value, step_value = sess.run(
            [train_op, variable, global_step], feed_dict={gradient: grad})
        self.assertAllEqual(variable_value, expected_variable)
        self.assertAllEqual(step_value, expected_step)

      _check_step([3.0, 3.0], [1.0, 2.0], 0)     # accum_grad = [3.0, 3.0]
      _check_step([4.0, 1.0], [1.0, 2.0], 0)     # accum_grad = [7.0, 4.0]
      _check_step([-1.0, 0.0], [-5.0, -2.0], 1)  # accum_grad = [6.0, 4.0], apply
      _check_step([-3.0, 1.0], [-5.0, -2.0], 1)  # accum_grad = [-3.0, 1.0]
      _check_step([0.0, -3.0], [-5.0, -2.0], 1)  # accum_grad = [-3.0, -2.0]
      _check_step([2.0, -1.0], [-4.0, 1.0], 2)   # accum_grad = [-1.0, -3.0], apply

  def testDelayedUpdateSparseGradients(self):
    # Test that delayed update does not crash on sparse gradients.
    global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
    optimizer = tf.train.GradientDescentOptimizer(1.0)
    embeddings = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
    x = tf.nn.embedding_lookup(embeddings, [0])
    loss = tf.losses.mean_squared_error([[1.1, 2.1]], x)
    gradients = optimizer.compute_gradients(loss)
    _ = optim.delayed_update(
        optimizer,
        gradients,
        global_step,
        accum_count=3)

  def testDelayedUpdateOptimizerSlots(self):
    # Test that delayed update does not change any variable names, in particular
    # optimizer variables.
    def _create_variables(accum_count):
      global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
      optimizer = tf.train.AdamOptimizer(1.0)
      gradient = tf.placeholder(tf.float32, shape=[2])
      variable = tf.Variable([1.0, 2.0])
      optim.delayed_update(
          optimizer,
          [(gradient, variable)],
          global_step,
          accum_count=accum_count)
      return list(sorted(var.name for var in tf.global_variables()))

    vars_no_accum = _create_variables(accum_count=1)
    tf.reset_default_graph()
    vars_accum = _create_variables(accum_count=3)
    self.assertListEqual(vars_accum, vars_no_accum)

if __name__ == "__main__":
  tf.test.main()
