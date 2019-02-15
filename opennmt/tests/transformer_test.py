import tensorflow as tf
import numpy as np

from opennmt.layers import transformer
from opennmt.tests import test_util


class TransformerTest(tf.test.TestCase):

  def testTileSequenceLength(self):
    num_heads = 2
    length = [5, 3, 7]
    tiled_length = transformer.tile_sequence_length(length, num_heads)
    tiled_length = self.evaluate(tiled_length)
    self.assertAllEqual([5, 5, 3, 3, 7, 7], tiled_length)

  def testBuildSequenceMask(self):
    num_heads = 4
    length = [5, 3, 7]
    expected = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    mask = transformer.build_sequence_mask(tf.constant(length), num_heads=num_heads)
    mask = self.evaluate(mask)
    self.assertTupleEqual(mask.shape, (len(length), 1, 1, max(length)))
    self.assertAllEqual(np.squeeze(mask), expected)

  def testBuildSequenceMaskWithMaxLen(self):
    num_heads = 4
    length = [5, 3, 6]
    maximum_length = 7
    expected = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]

    mask = transformer.build_sequence_mask(
        tf.constant(length), num_heads=num_heads, maximum_length=maximum_length)
    mask = self.evaluate(mask)
    self.assertTupleEqual(mask.shape, (len(length), 1, 1, maximum_length))
    self.assertAllEqual(np.squeeze(mask), expected)

  def testBuildFutureMask(self):
    num_heads = 4
    length = [2, 4, 3]
    expected = [
        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0, 1.0]],
        [[1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0, 0.0]]]

    mask = transformer.build_future_mask(tf.constant(length), num_heads=num_heads)
    mask = self.evaluate(mask)
    self.assertTupleEqual(mask.shape, (len(length), 1, max(length), max(length)))
    self.assertAllEqual(np.squeeze(mask), expected)

  def testBuildFutureMaskWithMaxLen(self):
    num_heads = 4
    length = [2, 4, 3]
    maximum_length = 5
    expected = [
        [[1.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 0.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0],
         [1.0, 1.0, 1.0, 0.0, 0.0]]]

    mask = transformer.build_future_mask(
        tf.constant(length), num_heads=num_heads, maximum_length=maximum_length)
    mask = self.evaluate(mask)
    self.assertTupleEqual(mask.shape, (len(length), 1, maximum_length, maximum_length))
    self.assertAllEqual(np.squeeze(mask), expected)

  def testCumulativeAverageMask(self):
    sequence_length = [2, 3]
    expected = [
        [[    1.0,     0.0, 0.0],
         [1.0/2.0, 1.0/2.0, 0.0],
         [    0.0,     0.0, 0.0]],
        [[    1.0,     0.0,     0.0],
         [1.0/2.0, 1.0/2.0,     0.0],
         [1.0/3.0, 1.0/3.0, 1.0/3.0]]]

    mask = transformer.cumulative_average_mask(tf.constant(sequence_length))
    mask = self.evaluate(mask)
    self.assertAllClose(expected, mask)

  def testCumulativeAverageMaskWithMaxLen(self):
    sequence_length = [2, 3]
    maximum_length = 4
    expected = [
        [[    1.0,     0.0, 0.0, 0.0],
         [1.0/2.0, 1.0/2.0, 0.0, 0.0],
         [    0.0,     0.0, 0.0, 0.0],
         [    0.0,     0.0, 0.0, 0.0]],
        [[    1.0,     0.0,     0.0, 0.0],
         [1.0/2.0, 1.0/2.0,     0.0, 0.0],
         [1.0/3.0, 1.0/3.0, 1.0/3.0, 0.0],
         [    0.0,     0.0,     0.0, 0.0]]]

    mask = transformer.cumulative_average_mask(
        tf.constant(sequence_length), maximum_length=maximum_length)
    mask = self.evaluate(mask)
    self.assertAllClose(expected, mask)

  def testSplitHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.convert_to_tensor(
        np.random.randn(batch_size, max(length), depth * num_heads).astype(np.float32))
    outputs = transformer.split_heads(inputs, num_heads)

    static_shape = outputs.get_shape().as_list()
    self.assertEqual(num_heads, static_shape[1])
    self.assertEqual(depth, static_shape[-1])
    outputs = self.evaluate(outputs)
    self.assertAllEqual([batch_size, num_heads, max(length), depth], outputs.shape)

  def testCombineHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.convert_to_tensor(
        np.random.randn(batch_size, num_heads, max(length), depth).astype(np.float32))
    outputs = transformer.combine_heads(inputs)

    static_shape = outputs.get_shape().as_list()
    self.assertEqual(depth * num_heads, static_shape[-1])
    outputs = self.evaluate(outputs)
    self.assertAllEqual([batch_size, max(length), depth * num_heads], outputs.shape)

  def testSplitAndCombineHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.convert_to_tensor(
        np.random.randn(batch_size, max(length), depth * num_heads).astype(np.float32))
    split = transformer.split_heads(inputs, num_heads)
    combined = transformer.combine_heads(split)
    inputs, combined = self.evaluate([inputs, combined])
    self.assertAllEqual(inputs, combined)

  @test_util.run_tf1_only
  def testScaledDotAttention(self):
    batch_size = 3
    num_heads = 8
    values_length = [5, 3, 7]
    queries_length = [8, 6, 10]
    depth = 20

    queries = tf.placeholder_with_default(
        np.random.randn(batch_size, num_heads, max(queries_length), depth).astype(np.float32),
        shape=(None, num_heads, None, depth))
    values = tf.placeholder_with_default(
        np.random.randn(batch_size, num_heads, max(values_length), depth).astype(np.float32),
        shape=(None, num_heads, None, depth))
    keys = values

    mask = transformer.build_sequence_mask(values_length, num_heads=num_heads)
    context, attn = transformer.dot_product_attention(
        queries,
        keys,
        values,
        tf.estimator.ModeKeys.PREDICT,
        mask=mask)

    with self.test_session() as sess:
      context, attn = sess.run([context, attn])
      self.assertTupleEqual(
          (batch_size, num_heads, max(queries_length), depth), context.shape)
      self.assertTupleEqual(
          (batch_size, num_heads, max(queries_length), max(values_length)), attn.shape)

      for i in range(batch_size):
        length = values_length[i]
        padding_length = max(values_length) - length
        if padding_length > 0:
          self.assertEqual(0.0, np.sum(attn[i, :, :, length:max(values_length)]))

  @test_util.run_tf1_only
  def testMaskedScaledDotAttention(self):
    batch_size = 3
    num_heads = 8
    queries_length = [8, 6, 10]
    depth = 20

    queries = tf.placeholder_with_default(
        np.random.randn(batch_size, num_heads, max(queries_length), depth).astype(np.float32),
        shape=(None, num_heads, None, depth))

    mask = transformer.build_future_mask(queries_length, num_heads=num_heads)
    context, attn = transformer.dot_product_attention(
        queries,
        queries,
        queries,
        tf.estimator.ModeKeys.PREDICT,
        mask=mask)

    with self.test_session() as sess:
      context, attn = sess.run([context, attn])
      illegal_connections = np.triu_indices(max(queries_length), 1)
      for i in range(batch_size):
        for h in range(num_heads):
          self.assertEqual(0.0, np.sum(attn[i, h][illegal_connections]))

  def testCumulativeAverage(self):
    x = [
      [[1.0], [2.0], [3.0], [0.0]],
      [[2.0], [4.0], [6.0], [8.0]]]
    y = [
      [[1.0], [(1.0+2.0)/2.0], [(1.0+2.0+3.0)/3.0], [0.0]],
      [[2.0], [(2.0+4.0)/2.0], [(2.0+4.0+6.0)/3.0], [(2.0+4.0+6.0+8.0)/4.0]]]
    lengths = [3, 4]

    mask = transformer.cumulative_average_mask(tf.constant(lengths))
    aa = transformer.cumulative_average(x, mask)
    aa = self.evaluate(aa)
    self.assertAllClose(y, aa)

  def testCumulativeAverageWithCache(self):
    x = tf.constant([
      [[1.0], [2.0], [3.0], [0.0]],
      [[2.0], [4.0], [6.0], [8.0]]])
    y = [
      [[1.0], [(1.0+2.0)/2.0], [(1.0+2.0+3.0)/3.0], [(1.0+2.0+3.0+0.0)/4.0]],
      [[2.0], [(2.0+4.0)/2.0], [(2.0+4.0+6.0)/3.0], [(2.0+4.0+6.0+8.0)/4.0]]]

    batch_size = tf.shape(x)[0]
    depth = x.get_shape().as_list()[-1]

    step = tf.constant(0)
    aa_ta = tf.TensorArray(tf.float32, size=tf.shape(x)[1])
    cache = {"prev_g": tf.zeros([batch_size, 1, depth], dtype=tf.float32)}

    def _cond(i, accu, cache):
      return i < tf.shape(x)[1]
    def _body(i, accu, cache):
      aa = transformer.cumulative_average(x[:, i:i+1], i, cache)
      return i + 1, accu.write(i, tf.squeeze(aa, axis=1)), cache

    _, aa_ta, _ = tf.while_loop(
        _cond,
        _body,
        loop_vars=(step, aa_ta, cache),
        shape_invariants=(
            tf.TensorShape([]),
            tf.TensorShape(None),
            {"prev_g": tf.TensorShape([None, None, depth])}),
        parallel_iterations=1)
    aa = tf.transpose(aa_ta.stack(), perm=(1, 0, 2))
    aa = self.evaluate(aa)
    self.assertAllClose(y, aa)

  @test_util.run_tf2_only
  def testFeedForwardNetwork(self):
    ffn = transformer.FeedForwardNetwork(20, 10)
    x = tf.random.uniform([4, 5, 10])
    y = ffn(x)
    self.assertEqual(y.shape, x.shape)

  @test_util.run_tf2_only
  def testMultiHeadSelfAttention(self):
    attention = transformer.MultiHeadAttention(4, 20)
    queries = tf.random.uniform([4, 5, 10])
    mask = tf.expand_dims(tf.sequence_mask([4, 3, 5, 2]), 1)
    context, _ = attention(queries, mask=mask)
    self.assertListEqual(context.shape.as_list(), [4, 5, 20])

  @test_util.run_tf2_only
  def testMultiHeadSelfAttentionWithCache(self):
    cache = (tf.zeros([4, 4, 0, 5]), tf.zeros([4, 4, 0, 5]))
    attention = transformer.MultiHeadAttention(4, 20)
    x = tf.random.uniform([4, 1, 10])
    _, cache = attention(x, cache=cache)
    self.assertEqual(cache[0].shape[2], 1)
    self.assertEqual(cache[1].shape[2], 1)
    _, cache = attention(x, cache=cache)
    self.assertEqual(cache[0].shape[2], 2)
    self.assertEqual(cache[1].shape[2], 2)

  @test_util.run_tf2_only
  def testMultiHeadAttention(self):
    attention = transformer.MultiHeadAttention(4, 20)
    queries = tf.random.uniform([4, 5, 10])
    memory = tf.random.uniform([4, 3, 10])
    mask = tf.expand_dims(tf.sequence_mask([1, 3, 2, 2]), 1)
    context, _ = attention(queries, memory=memory, mask=mask)
    self.assertListEqual(context.shape.as_list(), [4, 5, 20])

  @test_util.run_tf2_only
  def testMultiHeadAttentionWithCache(self):
    cache = (tf.zeros([4, 4, 0, 5]), tf.zeros([4, 4, 0, 5]))
    attention = transformer.MultiHeadAttention(4, 20)
    memory = tf.random.uniform([4, 3, 10])
    mask = tf.expand_dims(tf.sequence_mask([1, 3, 2, 2]), 1)
    x = tf.random.uniform([4, 1, 10])
    y1, cache = attention(x, memory=memory, mask=mask, cache=cache)
    self.assertEqual(cache[0].shape[2], 3)
    self.assertEqual(cache[1].shape[2], 3)
    y2, cache = attention(x, memory=memory, mask=mask, cache=cache)
    self.assertAllEqual(y1, y2)

  @test_util.run_tf2_only
  def testMultiHeadAttentionMask(self):
    attention = transformer.MultiHeadAttention(4, 20, return_attention=True)
    queries = tf.random.uniform([4, 5, 10])
    memory = tf.random.uniform([4, 3, 10])
    mask = tf.expand_dims(tf.sequence_mask([1, 3, 2, 2]), 1)
    _, _, attention = attention(queries, memory=memory, mask=mask)
    attention = tf.reshape(attention, [4, -1, 3])
    mask = tf.broadcast_to(mask, attention.shape)
    padding = tf.boolean_mask(attention, tf.logical_not(mask))
    self.assertAllEqual(tf.reduce_sum(padding), 0)


if __name__ == "__main__":
  tf.test.main()
