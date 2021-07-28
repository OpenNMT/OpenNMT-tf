import numpy as np
import tensorflow as tf

from parameterized import parameterized

from opennmt.layers import transformer


class TransformerTest(tf.test.TestCase):
    @parameterized.expand([[tf.bool], [tf.float32]])
    def testBuildFutureMask(self, dtype):
        length = [2, 4, 3]
        expected = np.array(
            [
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 0]],
            ]
        ).astype(dtype.as_numpy_dtype)

        mask = transformer.future_mask(tf.constant(length), dtype=dtype)
        self.assertIs(mask.dtype, dtype)
        mask = self.evaluate(mask)
        self.assertTupleEqual(mask.shape, (len(length), max(length), max(length)))
        self.assertAllEqual(mask, expected)

    @parameterized.expand([[tf.bool], [tf.float32]])
    def testBuildFutureMaskWithMaxLen(self, dtype):
        length = [2, 4, 3]
        maximum_length = 5
        expected = np.array(
            [
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                ],
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 0],
                ],
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 0, 0],
                ],
            ]
        ).astype(dtype.as_numpy_dtype)

        mask = transformer.future_mask(
            tf.constant(length), maximum_length=maximum_length, dtype=dtype
        )
        self.assertIs(mask.dtype, dtype)
        mask = self.evaluate(mask)
        self.assertTupleEqual(mask.shape, (len(length), maximum_length, maximum_length))
        self.assertAllEqual(mask, expected)

    def testSplitHeads(self):
        batch_size = 3
        length = [5, 3, 7]
        num_heads = 8
        depth = 20

        inputs = tf.random.normal(
            [batch_size, max(length), depth * num_heads], dtype=tf.float32
        )
        outputs = transformer.split_heads(inputs, num_heads)

        static_shape = outputs.shape
        self.assertEqual(num_heads, static_shape[1])
        self.assertEqual(depth, static_shape[-1])
        outputs = self.evaluate(outputs)
        self.assertAllEqual([batch_size, num_heads, max(length), depth], outputs.shape)

    def testCombineHeads(self):
        batch_size = 3
        length = [5, 3, 7]
        num_heads = 8
        depth = 20

        inputs = tf.random.normal(
            [batch_size, num_heads, max(length), depth], dtype=tf.float32
        )
        outputs = transformer.combine_heads(inputs)

        static_shape = outputs.shape
        self.assertEqual(depth * num_heads, static_shape[-1])
        outputs = self.evaluate(outputs)
        self.assertAllEqual([batch_size, max(length), depth * num_heads], outputs.shape)

    def testSplitAndCombineHeads(self):
        batch_size = 3
        length = [5, 3, 7]
        num_heads = 8
        depth = 20

        inputs = tf.random.normal(
            [batch_size, max(length), depth * num_heads], dtype=tf.float32
        )
        split = transformer.split_heads(inputs, num_heads)
        combined = transformer.combine_heads(split)
        inputs, combined = self.evaluate([inputs, combined])
        self.assertAllEqual(inputs, combined)

    def testRelativePositions(self):
        positions = transformer.relative_positions(4, 2)
        self.assertAllEqual(
            self.evaluate(positions),
            [[2, 3, 4, 4], [1, 2, 3, 4], [0, 1, 2, 3], [0, 0, 1, 2]],
        )

    def testFeedForwardNetwork(self):
        ffn = transformer.FeedForwardNetwork(20, 10)
        x = tf.random.uniform([4, 5, 10])
        y = ffn(x)
        self.assertEqual(y.shape, x.shape)

    def testMultiHeadSelfAttention(self):
        attention = transformer.MultiHeadAttention(4, 20)
        queries = tf.random.uniform([4, 5, 10])
        mask = tf.sequence_mask([4, 3, 5, 2])
        context, _ = attention(queries, mask=mask)
        self.assertListEqual(context.shape.as_list(), [4, 5, 20])

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

    def testMultiHeadSelfAttentionRelativePositions(self):
        attention = transformer.MultiHeadAttention(4, 20, maximum_relative_position=6)
        x = tf.random.uniform([2, 9, 10])
        mask = tf.sequence_mask([9, 7])
        attention(x, mask=mask)

    def testMultiHeadSelfAttentionRelativePositionsEmpty(self):
        attention = transformer.MultiHeadAttention(4, 20, maximum_relative_position=6)
        x = tf.random.uniform([1, 0, 10])
        mask = tf.sequence_mask([0])
        y, _ = attention(x, mask=mask)
        self.assertListEqual(y.shape.as_list(), [1, 0, 20])

    def testMultiHeadSelfAttentionRelativePositionsWithCache(self):
        attention = transformer.MultiHeadAttention(4, 20, maximum_relative_position=6)
        x = tf.random.uniform([4, 1, 10])
        cache = (tf.zeros([4, 4, 0, 5]), tf.zeros([4, 4, 0, 5]))
        _, cache = attention(x, cache=cache)

    def testMultiHeadSelfAttentionRelativeGradients(self):
        attention = transformer.MultiHeadAttention(4, 20, maximum_relative_position=6)

        @tf.function
        def _compute_gradients_in_function(x):
            with tf.GradientTape() as tape:
                y, _ = attention(x)
                loss = tf.math.reduce_sum(y)
            gradients = tape.gradient(loss, attention.weights)
            for gradient in gradients:
                self.assertTrue(gradient.shape.is_fully_defined())

        _compute_gradients_in_function(tf.random.uniform([4, 1, 10]))

    def testMultiHeadAttention(self):
        attention = transformer.MultiHeadAttention(4, 20, return_attention=True)
        queries = tf.random.uniform([4, 5, 10])
        memory = tf.random.uniform([4, 3, 10])
        mask = tf.sequence_mask([1, 3, 2, 2])
        context, _, attention = attention(queries, memory=memory, mask=mask)
        self.assertListEqual(context.shape.as_list(), [4, 5, 20])
        self.assertListEqual(attention.shape.as_list(), [4, 4, 5, 3])

    def testMultiHeadAttentionWithCache(self):
        cache = (tf.zeros([4, 4, 0, 5]), tf.zeros([4, 4, 0, 5]))
        attention = transformer.MultiHeadAttention(4, 20)
        memory = tf.random.uniform([4, 3, 10])
        mask = tf.sequence_mask([1, 3, 2, 2])
        x = tf.random.uniform([4, 1, 10])
        y1, cache = attention(x, memory=memory, mask=mask, cache=cache)
        self.assertEqual(cache[0].shape[2], 3)
        self.assertEqual(cache[1].shape[2], 3)
        y2, cache = attention(x, memory=memory, mask=mask, cache=cache)
        self.assertAllEqual(y1, y2)

    def testMultiHeadAttentionMask(self):
        attention = transformer.MultiHeadAttention(4, 20, return_attention=True)
        queries = tf.random.uniform([4, 5, 10])
        memory = tf.random.uniform([4, 3, 10])
        mask = tf.sequence_mask([1, 3, 2, 2])
        _, _, attention = attention(queries, memory=memory, mask=mask)
        attention = tf.reshape(attention, [4, -1, 3])
        mask = tf.broadcast_to(tf.expand_dims(mask, 1), attention.shape)
        padding = tf.boolean_mask(attention, tf.logical_not(mask))
        self.assertAllEqual(tf.reduce_sum(padding), 0)


if __name__ == "__main__":
    tf.test.main()
