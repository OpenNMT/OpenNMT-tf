from parameterized import parameterized

import tensorflow as tf

from opennmt.utils import optim


class OptimTest(tf.test.TestCase):

  @parameterized.expand([
      ["l1", 1e-4],
      ["L1", 1e-4],
      ["l1", 1],
      ["l2", 1e-4],
      ["l1_l2", (1e-4, 1e-4)],
  ])
  def testRegularization(self, type, scale):
    layer = tf.keras.layers.Dense(256)
    layer.build([None, 128])
    regularization = optim.regularization_penalty(
        type, scale, layer.trainable_variables)
    self.assertEqual(0, len(regularization.shape.as_list()))
    self.evaluate(regularization)

  def testRegulaizationInvalidType(self):
    with self.assertRaises(ValueError):
      optim.regularization_penalty("l3", 1e-4, [])

  def testRegulaizationMissingScaleValue(self):
    with self.assertRaises(ValueError):
      optim.regularization_penalty("l1_l2", 1e-4, [])

  def testDelayedUpdate(self):
    with tf.Graph().as_default():
      optimizer = tf.optimizers.SGD(1.0)
      gradient = tf.compat.v1.placeholder(tf.float32, shape=[2])
      variable = tf.Variable([1.0, 2.0])
      train_op, extra_variables = optim.delayed_update(
          optimizer,
          [(gradient, variable)],
          accum_count=3)
      with self.session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        sess.run(tf.compat.v1.variables_initializer(extra_variables))

        def _check_step(grad, expected_variable):
          sess.run(train_op, feed_dict={gradient: grad})
          self.assertAllEqual(sess.run(variable), expected_variable)

        _check_step([3.0, 3.0], [1.0, 2.0])     # accum_grad = [3.0, 3.0]
        _check_step([4.0, 1.0], [1.0, 2.0])     # accum_grad = [7.0, 4.0]
        _check_step([-1.0, 0.0], [-5.0, -2.0])  # accum_grad = [6.0, 4.0], apply
        _check_step([-3.0, 1.0], [-5.0, -2.0])  # accum_grad = [-3.0, 1.0]
        _check_step([0.0, -3.0], [-5.0, -2.0])  # accum_grad = [-3.0, -2.0]
        _check_step([2.0, -1.0], [-4.0, 1.0])   # accum_grad = [-1.0, -3.0], apply

  def testDelayedUpdateSparseGradients(self):
    with tf.Graph().as_default():
      # Test that delayed update does not crash on sparse gradients.
      optimizer = tf.optimizers.SGD(1.0)
      embeddings = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
      x = tf.nn.embedding_lookup(embeddings, [0])
      loss = tf.losses.MeanSquaredError()([[1.1, 2.1]], x)
      gradients = optimizer.get_gradients(loss, embeddings)
      _ = optim.delayed_update(
          optimizer,
          zip(gradients, [embeddings]),
          accum_count=3)


if __name__ == "__main__":
  tf.test.main()
