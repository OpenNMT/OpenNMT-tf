import tensorflow as tf
import numpy as np

from opennmt.layers import position


class _DummyPositionEncoder(position.PositionEncoder):
  """Encoder that simply forwards the position indices."""

  def encode(self, positions, depth, dtype=tf.float32):
    positions = tf.expand_dims(positions, 2)
    positions = tf.tile(positions, [1, 1, depth])
    return tf.cast(positions, dtype)


class PositionTest(tf.test.TestCase):

  def testPositionBuilder(self):
    sequence_length = tf.constant([4, 6])
    positions = position.make_positions(sequence_length)
    positions = self.evaluate(positions)
    self.assertAllEqual([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 5, 6]], positions)

  def testPositionBuilderWithMaxLen(self):
    sequence_length = tf.constant([4, 6])
    positions = position.make_positions(sequence_length, maximum_length=7)
    positions = self.evaluate(positions)
    self.assertAllEqual([[1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 5, 6, 0]], positions)

  def testApplyOneEncoding(self):
    encoder = _DummyPositionEncoder()
    inputs = tf.zeros([2, 1, 3])
    outputs = encoder.apply_one(inputs, 2)
    outputs = self.evaluate(outputs)
    self.assertAllEqual(outputs, [[[2, 2, 2]], [[2, 2, 2]]])

  def testApplyPositionEncoding(self):
    encoder = _DummyPositionEncoder()
    sequence_length = tf.constant([2, 3])
    inputs = tf.zeros([2, 4, 3])
    outputs = encoder.apply(inputs, sequence_length=sequence_length)
    outputs = self.evaluate(outputs)
    self.assertAllEqual(outputs, [
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    ])

  def testApplyPositionEncodingWithoutSequenceLength(self):
    encoder = _DummyPositionEncoder()
    inputs = tf.zeros([2, 4, 3])
    outputs = encoder.apply(inputs)
    outputs = self.evaluate(outputs)
    self.assertAllEqual(outputs, [
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]
    ])

  def _testSinusoidalPositionEncoder(self, depth, dtype=tf.float32):
    encoder = position.SinusoidalPositionEncoder()
    positions = position.make_positions([4, 6])
    encoding = encoder.encode(positions, depth, dtype=dtype)
    self.assertEqual(dtype, encoding.dtype.base_dtype)
    encoding = self.evaluate(encoding)
    self.assertAllEqual([2, 6, depth], encoding.shape)

  def testSinusoidalPositionEncoder(self):
    self._testSinusoidalPositionEncoder(10)
  def testSinusoidalPositionEncoderFloat16(self):
    self._testSinusoidalPositionEncoder(10, dtype=tf.float16)
  def testSinusoidalPositionEncoderInvalidDepth(self):
    with self.assertRaises(ValueError):
      self._testSinusoidalPositionEncoder(5)


if __name__ == "__main__":
  tf.test.main()
