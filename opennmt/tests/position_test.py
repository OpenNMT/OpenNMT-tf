import tensorflow as tf

from parameterized import parameterized

from opennmt.layers import position


class _DummyPositionEncoder(position.PositionEncoder):
    """Encoder that simply forwards the position indices."""

    def _encode(self, positions, depth):
        positions = tf.expand_dims(positions, 2)
        positions = tf.tile(positions, [1, 1, depth])
        return tf.cast(positions, self.dtype)


class PositionTest(tf.test.TestCase):
    def testApplyOneEncoding(self):
        encoder = _DummyPositionEncoder()
        inputs = tf.zeros([2, 1, 3])
        outputs = encoder(inputs, 2)
        outputs = self.evaluate(outputs)
        self.assertAllEqual(outputs, [[[2, 2, 2]], [[2, 2, 2]]])

    def testApplyPositionEncoding(self):
        encoder = _DummyPositionEncoder()
        inputs = tf.zeros([2, 4, 3])
        outputs = encoder(inputs)
        outputs = self.evaluate(outputs)
        self.assertAllEqual(
            outputs,
            [
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
                [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            ],
        )

    def _testSinusoidalPositionEncoder(self, depth, dtype=tf.float32):
        encoder = position.SinusoidalPositionEncoder(dtype=dtype)
        inputs = tf.zeros([2, 6, depth], dtype=dtype)
        outputs = encoder(inputs)
        self.assertEqual(dtype, outputs.dtype.base_dtype)
        outputs = self.evaluate(outputs)
        self.assertAllEqual([2, 6, depth], outputs.shape)

    def testSinusoidalPositionEncoder(self):
        self._testSinusoidalPositionEncoder(10)

    def testSinusoidalPositionEncoderFloat16(self):
        self._testSinusoidalPositionEncoder(10, dtype=tf.float16)

    def testSinusoidalPositionEncoderInvalidDepth(self):
        with self.assertRaises(ValueError):
            self._testSinusoidalPositionEncoder(5)

    @parameterized.expand([[tf.float32], [tf.float16]])
    def testPositionEmbedder(self, dtype):
        encoder = position.PositionEmbedder(dtype=dtype)
        inputs = tf.zeros([3, 5, 10], dtype=dtype)
        outputs = encoder(inputs)
        self.assertEqual(outputs.dtype, dtype)
        self.assertEqual(encoder.embedding.dtype.base_dtype, dtype)
        outputs = self.evaluate(outputs)
        self.assertAllEqual(outputs.shape, [3, 5, 10])


if __name__ == "__main__":
    tf.test.main()
