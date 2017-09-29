import tensorflow as tf
import numpy as np

from opennmt.utils import transformer


class TransformerTest(tf.test.TestCase):

  def testScaledDotAttention(self):
    batch_size = 3
    values_length = [5, 3, 7]
    queries_length = [8, 6, 10]
    depth = 20

    queries = tf.placeholder_with_default(
      np.random.randn(batch_size, max(queries_length), depth).astype(np.float32),
      shape=(None, None, depth))
    values = tf.placeholder_with_default(
      np.random.randn(batch_size, max(values_length), depth).astype(np.float32),
      shape=(None, None, depth))
    keys = values

    context, attn = transformer.scaled_dot_attention(
        queries,
        keys,
        values,
        tf.estimator.ModeKeys.PREDICT,
        values_length=tf.constant(values_length))

    with self.test_session() as sess:
      context, attn = sess.run([context, attn])
      self.assertTupleEqual(context.shape, (batch_size, max(queries_length), depth))
      self.assertTupleEqual(attn.shape, (batch_size, max(queries_length), max(values_length)))

      for i in range(batch_size):
        length = values_length[i]
        padding_length = max(values_length) - length
        if padding_length > 0:
          self.assertEqual(0.0, np.sum(attn[i,:,length:max(values_length)]))

  def testMaskedScaledDotAttention(self):
    batch_size = 3
    queries_length = [8, 6, 10]
    depth = 20

    queries = tf.placeholder_with_default(
      np.random.randn(batch_size, max(queries_length), depth).astype(np.float32),
      shape=(None, None, depth))

    context, attn = transformer.scaled_dot_attention(
        queries,
        queries,
        queries,
        tf.estimator.ModeKeys.PREDICT,
        values_length=tf.constant(queries_length),
        mask_future=True)

    with self.test_session() as sess:
      context, attn = sess.run([context, attn])
      illegal_connections = np.triu_indices(max(queries_length), 1)
      for i in range(batch_size):
        self.assertEqual(0.0, np.sum(attn[i][illegal_connections]))



if __name__ == '__main__':
  tf.test.main()
