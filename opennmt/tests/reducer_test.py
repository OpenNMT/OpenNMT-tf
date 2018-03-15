import collections

import tensorflow as tf

from opennmt.layers import reducer


class ReducerTest(tf.test.TestCase):

  def testPadWithIdentity(self):
    tensor = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    expected = [
        [[1], [1], [1], [1], [0]],
        [[1], [2], [3], [1], [1]],
        [[1], [2], [0], [0], [0]]]
    lengths = [1, 3, 2]
    max_lengths = [4, 5, 2]

    padded = reducer.pad_with_identity(
        tf.constant(tensor, dtype=tf.float32),
        tf.constant(lengths),
        tf.constant(max_lengths),
        identity_values=1)

    self.assertEqual(1, padded.get_shape().as_list()[-1])

    with self.test_session() as sess:
      padded = sess.run(padded)
      self.assertAllEqual(expected, padded)

  def testPadNWithIdentity(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4]],
        [[1], [2], [-1], [-1]],
        [[1], [2], [-1], [-1]]]
    expected_a = [
        [[1], [1], [1], [1]],
        [[1], [2], [3], [0]],
        [[1], [2], [0], [0]]]
    expected_b = [
        [[1], [2], [3], [4]],
        [[1], [2], [1], [0]],
        [[1], [2], [0], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    (padded_a, padded_b), length = reducer.pad_n_with_identity(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)],
        identity_values=1)

    with self.test_session() as sess:
      padded_a, padded_b, length = sess.run([padded_a, padded_b, length])
      self.assertAllEqual([4, 3, 2], length)
      self.assertAllEqual(expected_a, padded_a)
      self.assertAllEqual(expected_b, padded_b)

  def testRollSequence(self):
    offset = [2, 3, 3]
    tensor = [
        [1, 2, 3, 0, 0, 6, 0],
        [1, 2, 3, 4, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 7]]
    expected = [
        [6, 0, 1, 2, 3, 0, 0],
        [0, 0, 0, 1, 2, 3, 4],
        [0, 0, 7, 1, 0, 0, 0]]

    rolled = reducer.roll_sequence(tensor, offset)

    with self.test_session() as sess:
      self.assertAllEqual(expected, sess.run(rolled))

  def testMultiplyReducerWithSequence(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4]],
        [[1], [2], [-1], [-1]],
        [[1], [2], [-1], [-1]]]
    expected = [
        [[1], [2], [3], [4]],
        [[1], [4], [3], [0]],
        [[1], [4], [0], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    reduced, length = reducer.MultiplyReducer().reduce_sequence(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    with self.test_session() as sess:
      reduced, length = sess.run([reduced, length])
      self.assertAllEqual(expected, reduced)
      self.assertAllEqual([4, 3, 2], length)

  def testConcatInDepthWithSequence(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4]],
        [[1], [2], [-1], [-1]],
        [[1], [2], [-1], [-1]]]
    expected = [
        [[1, 1], [0, 2], [0, 3], [0, 4]],
        [[1, 1], [2, 2], [3, 0], [0, 0]],
        [[1, 1], [2, 2], [0, 0], [0, 0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    reduced, length = reducer.ConcatReducer().reduce_sequence(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    self.assertEqual(2, reduced.get_shape().as_list()[-1])

    with self.test_session() as sess:
      reduced, length = sess.run([reduced, length])
      self.assertAllEqual(expected, reduced)
      self.assertAllEqual([4, 3, 2], length)

  def testConcatInTimeWithSequence(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4]],
        [[1], [2], [-1], [-1]],
        [[1], [2], [-1], [-1]]]
    expected = [
        [[1], [1], [2], [3], [4]],
        [[1], [2], [3], [1], [2]],
        [[1], [2], [1], [2], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    reduced, length = reducer.ConcatReducer(axis=1).reduce_sequence(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    self.assertEqual(1, reduced.get_shape().as_list()[-1])

    with self.test_session() as sess:
      reduced, length = sess.run([reduced, length])
      self.assertAllEqual(expected, reduced)
      self.assertAllEqual([5, 5, 4], length)

  def testJoinReducer(self):
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer().reduce([1, 2, 3]))
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer().reduce([(1,), (2,), (3,)]))
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer().reduce([1, (2, 3)]))

    # Named tuples should not be unpacked.
    State = collections.namedtuple("State", ["h", "c"])
    self.assertTupleEqual((State(h=1, c=2), State(h=3, c=4), State(h=5, c=6)),
                          reducer.JoinReducer().reduce([
                              State(h=1, c=2), (State(h=3, c=4), State(h=5, c=6))]))


if __name__ == "__main__":
  tf.test.main()
