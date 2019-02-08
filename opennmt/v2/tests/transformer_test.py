from parameterized import parameterized

import tensorflow as tf

from opennmt.v2.layers import transformer


class TransformerTest(tf.test.TestCase):

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testFeedForwardNetwork(self, dtype):
    ffn = transformer.FeedForwardNetwork(20, 10)
    x = tf.random.uniform([4, 5, 10], dtype=dtype)
    y = ffn(x)
    self.assertEqual(y.shape, x.shape)
    self.assertEqual(y.dtype, dtype)

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testMultiHeadSelfAttention(self, dtype):
    attention = transformer.MultiHeadAttention(4, 20)
    queries = tf.random.uniform([4, 5, 10], dtype=dtype)
    mask = tf.expand_dims(tf.sequence_mask([4, 3, 5, 2]), 1)
    context, _ = attention(queries, mask=mask)
    self.assertListEqual(context.shape.as_list(), [4, 5, 20])
    self.assertEqual(context.dtype, dtype)

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testMultiHeadSelfAttentionWithCache(self, dtype):
    cache = (tf.zeros([4, 4, 0, 5], dtype=dtype), tf.zeros([4, 4, 0, 5], dtype=dtype))
    attention = transformer.MultiHeadAttention(4, 20)
    x = tf.random.uniform([4, 1, 10], dtype=dtype)
    _, cache = attention(x, cache=cache)
    self.assertEqual(cache[0].shape[2], 1)
    self.assertEqual(cache[1].shape[2], 1)
    _, cache = attention(x, cache=cache)
    self.assertEqual(cache[0].shape[2], 2)
    self.assertEqual(cache[0].dtype, dtype)
    self.assertEqual(cache[1].shape[2], 2)
    self.assertEqual(cache[1].dtype, dtype)

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testMultiHeadAttention(self, dtype):
    attention = transformer.MultiHeadAttention(4, 20)
    queries = tf.random.uniform([4, 5, 10], dtype=dtype)
    memory = tf.random.uniform([4, 3, 10], dtype=dtype)
    mask = tf.expand_dims(tf.sequence_mask([1, 3, 2, 2]), 1)
    context, _ = attention(queries, memory=memory, mask=mask)
    self.assertListEqual(context.shape.as_list(), [4, 5, 20])
    self.assertEqual(context.dtype, dtype)

  @parameterized.expand([[tf.float32], [tf.float16]])
  def testMultiHeadAttentionWithCache(self, dtype):
    cache = (tf.zeros([4, 4, 0, 5], dtype=dtype), tf.zeros([4, 4, 0, 5], dtype=dtype))
    attention = transformer.MultiHeadAttention(4, 20)
    memory = tf.random.uniform([4, 3, 10], dtype=dtype)
    mask = tf.expand_dims(tf.sequence_mask([1, 3, 2, 2]), 1)
    x = tf.random.uniform([4, 1, 10], dtype=dtype)
    y1, cache = attention(x, memory=memory, mask=mask, cache=cache)
    self.assertEqual(cache[0].shape[2], 3)
    self.assertEqual(cache[0].dtype, dtype)
    self.assertEqual(cache[1].shape[2], 3)
    self.assertEqual(cache[1].dtype, dtype)
    y2, cache = attention(x, memory=memory, mask=mask, cache=cache)
    self.assertAllEqual(y1, y2)


if __name__ == "__main__":
  tf.test.main()
