import tensorflow as tf

from opennmt.utils import position


class PositionTest(tf.test.TestCase):

  def testPositionBuilder(self):
    sequence_length = tf.constant([4, 6])
    positions = position.make_positions(sequence_length)
    with self.test_session() as sess:
      positions = sess.run(positions)
      self.assertAllEqual([[1, 2, 3, 4, 0, 0], [1, 2, 3, 4, 5, 6]], positions)


if __name__ == "__main__":
  tf.test.main()
