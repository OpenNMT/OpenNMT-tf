import tensorflow as tf

from opennmt.layers import position


class PositionTest(tf.test.TestCase):

  def testPositionBuilder(self):
    sequence_length = tf.constant([4, 6])
    positions = position.make_positions(sequence_length)
    with self.test_session() as sess:
      positions = sess.run(positions)
      self.assertAllEqual([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 5, 6]], positions)

  def testPositionBuilderWithMaxLen(self):
    sequence_length = tf.constant([4, 6])
    positions = position.make_positions(sequence_length, maximum_length=7)
    with self.test_session() as sess:
      positions = sess.run(positions)
      self.assertAllEqual([[1, 2, 3, 4, 0, 0, 0], [1, 2, 3, 4, 5, 6, 0]], positions)

  def _testSinusoidalPositionEncoder(self, depth):
    encoder = position.SinusoidalPositionEncoder()
    positions = position.make_positions([4, 6])
    encoding = encoder.encode(positions, depth)
    with self.test_session() as sess:
      encoding = sess.run(encoding)
      self.assertAllEqual([2, 6, depth], encoding.shape)

  def testSinusoidalPositionEncoder(self):
    self._testSinusoidalPositionEncoder(10)
  def testSinusoidalPositionEncoderInvalidDepth(self):
    with self.assertRaises(ValueError):
      self._testSinusoidalPositionEncoder(5)


if __name__ == "__main__":
  tf.test.main()
