import tensorflow as tf
import numpy as np

from opennmt.layers import transformer


class TransformerTest(tf.test.TestCase):

  def testTileSequenceLength(self):
    num_heads = 2
    length = [5, 3, 7]
    tiled_length = transformer.tile_sequence_length(length, num_heads)
    with self.test_session() as sess:
      tiled_length = sess.run(tiled_length)
      self.assertAllEqual([5, 5, 3, 3, 7, 7], tiled_length)

  def testBuildSequenceMask(self):
    num_heads = 4
    length = [5, 3, 7]
    expected = [
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]

    mask = transformer.build_sequence_mask(tf.constant(length), num_heads=num_heads)

    with self.test_session() as sess:
      mask = sess.run(mask)
      mask = np.reshape(mask, (len(length), num_heads, max(length)))
      mask = np.transpose(mask, (1, 0, 2))
      for b in range(len(length)):
        self.assertAllEqual(expected, mask[b])

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

    with self.test_session() as sess:
      mask = sess.run(mask)
      mask = np.reshape(mask, (len(length), num_heads, maximum_length))
      mask = np.transpose(mask, (1, 0, 2))
      for b in range(len(length)):
        self.assertAllEqual(expected, mask[b])

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

    with self.test_session() as sess:
      mask = sess.run(mask)
      mask = np.transpose(mask, (1, 0, 2, 3))
      for b in range(len(length)):
        self.assertAllEqual(expected, mask[b])

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

    with self.test_session() as sess:
      mask = sess.run(mask)
      mask = np.transpose(mask, (1, 0, 2, 3))
      for b in range(len(length)):
        self.assertAllEqual(expected, mask[b])

  def testSplitHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, max(length), depth * num_heads).astype(np.float32),
        shape=(None, None, depth * num_heads))
    outputs = transformer.split_heads(inputs, num_heads)

    static_shape = outputs.get_shape().as_list()
    self.assertEqual(num_heads, static_shape[1])
    self.assertEqual(depth, static_shape[-1])

    with self.test_session() as sess:
      outputs = sess.run(outputs)
      self.assertAllEqual([batch_size, num_heads, max(length), depth], outputs.shape)

  def testCombineHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, num_heads, max(length), depth).astype(np.float32),
        shape=(None, num_heads, None, depth))
    outputs = transformer.combine_heads(inputs)

    static_shape = outputs.get_shape().as_list()
    self.assertEqual(depth * num_heads, static_shape[-1])

    with self.test_session() as sess:
      outputs = sess.run(outputs)
      self.assertAllEqual([batch_size, max(length), depth * num_heads], outputs.shape)

  def testSplitAndCombineHeads(self):
    batch_size = 3
    length = [5, 3, 7]
    num_heads = 8
    depth = 20

    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, max(length), depth * num_heads).astype(np.float32),
        shape=(None, None, depth * num_heads))
    split = transformer.split_heads(inputs, num_heads)
    combined = transformer.combine_heads(split)

    with self.test_session() as sess:
      inputs, combined = sess.run([inputs, combined])
      self.assertAllEqual(inputs, combined)

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


if __name__ == "__main__":
  tf.test.main()
