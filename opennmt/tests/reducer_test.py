import collections

import tensorflow as tf

from parameterized import parameterized
from opennmt.layers import reducer


class ReducerTest(tf.test.TestCase):

  def testAlignInTimeSame(self):
    a = [
        [[1], [0], [0]],
        [[1], [2], [3]]]
    length = 3
    b = reducer.align_in_time(tf.constant(a, dtype=tf.float32), tf.constant(length))
    self.assertEqual(1, b.shape[-1])
    self.assertAllEqual(a, self.evaluate(b))

  def testAlignInTimeLarger(self):
    a = [
        [[1], [0], [0]],
        [[1], [2], [3]]]
    length = 4
    b = [
        [[1], [0], [0], [0]],
        [[1], [2], [3], [0]]]
    c = reducer.align_in_time(tf.constant(a, dtype=tf.float32), tf.constant(length))
    self.assertEqual(1, c.shape[-1])
    self.assertAllEqual(b, self.evaluate(c))

  def testAlignInTimeSmaller(self):
    a = [
        [[1], [0], [0]],
        [[1], [2], [0]]]
    length = 2
    b = [
        [[1], [0]],
        [[1], [2]]]
    c = reducer.align_in_time(tf.constant(a, dtype=tf.float32), tf.constant(length))
    self.assertEqual(1, c.shape[-1])
    self.assertAllEqual(b, self.evaluate(c))

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

    self.assertEqual(1, padded.shape[-1])
    self.assertAllEqual(expected, self.evaluate(padded))

  def testPadWithIdentityWithMaxTime(self):
    tensor = [
        [[1], [-1], [-1], [-1]],
        [[1], [2], [3], [-1]],
        [[1], [2], [-1], [-1]]]
    expected = [
        [[1], [1], [1], [1], [0], [0]],
        [[1], [2], [3], [1], [1], [0]],
        [[1], [2], [0], [0], [0], [0]]]
    lengths = [1, 3, 2]
    max_lengths = [4, 5, 2]
    maxlen = 6

    padded = reducer.pad_with_identity(
        tf.constant(tensor, dtype=tf.float32),
        tf.constant(lengths),
        tf.constant(max_lengths),
        identity_values=1,
        maxlen=maxlen)

    self.assertEqual(1, padded.shape[-1])
    self.assertAllEqual(expected, self.evaluate(padded))

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

    padded_a, padded_b, length = self.evaluate([padded_a, padded_b, length])
    self.assertAllEqual([4, 3, 2], length)
    self.assertAllEqual(expected_a, padded_a)
    self.assertAllEqual(expected_b, padded_b)

  def testPadNWithIdentityWithMaxTime(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4], [-1]],
        [[1], [2], [-1], [-1], [-1]],
        [[1], [2], [-1], [-1], [-1]]]
    expected_a = [
        [[1], [1], [1], [1], [0]],
        [[1], [2], [3], [0], [0]],
        [[1], [2], [0], [0], [0]]]
    expected_b = [
        [[1], [2], [3], [4], [0]],
        [[1], [2], [1], [0], [0]],
        [[1], [2], [0], [0], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    (padded_a, padded_b), length = reducer.pad_n_with_identity(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)],
        identity_values=1)

    padded_a, padded_b, length = self.evaluate([padded_a, padded_b, length])
    self.assertAllEqual([4, 3, 2], length)
    self.assertAllEqual(expected_a, padded_a)
    self.assertAllEqual(expected_b, padded_b)

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

    reduced, length = reducer.MultiplyReducer()(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    reduced, length = self.evaluate([reduced, length])
    self.assertAllEqual(expected, reduced)
    self.assertAllEqual([4, 3, 2], length)

  def testMultiplyReducerWithSequenceAndMaxTime(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4], [-1]],
        [[1], [2], [-1], [-1], [-1]],
        [[1], [2], [-1], [-1], [-1]]]
    expected = [
        [[1], [2], [3], [4], [0]],
        [[1], [4], [3], [0], [0]],
        [[1], [4], [0], [0], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    reduced, length = reducer.MultiplyReducer()(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    reduced, length = self.evaluate([reduced, length])
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

    reduced, length = reducer.ConcatReducer()(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    self.assertEqual(2, reduced.shape[-1])
    reduced, length = self.evaluate([reduced, length])
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

    reduced, length = reducer.ConcatReducer(axis=1)(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    self.assertEqual(1, reduced.shape[-1])
    reduced, length = self.evaluate([reduced, length])
    self.assertAllEqual(expected, reduced)
    self.assertAllEqual([5, 5, 4], length)

  def testConcatInTimeWithSequenceAndMaxTimeMismatch(self):
    a = [
        [[1], [-1], [-1]],
        [[1], [2], [3]],
        [[1], [2], [-1]]]
    b = [
        [[1], [2], [3], [4], [-1], [-1]],
        [[1], [2], [-1], [-1], [-1], [-1]],
        [[1], [2], [-1], [-1], [-1], [-1]]]
    expected = [
        [[1], [1], [2], [3], [4]],
        [[1], [2], [3], [1], [2]],
        [[1], [2], [1], [2], [0]]]
    length_a = [1, 3, 2]
    length_b = [4, 2, 2]

    reduced, length = reducer.ConcatReducer(axis=1)(
        [tf.constant(a, dtype=tf.float32), tf.constant(b, dtype=tf.float32)],
        [tf.constant(length_a), tf.constant(length_b)])

    self.assertEqual(1, reduced.shape[-1])
    reduced, length = self.evaluate([reduced, length])
    self.assertAllEqual(expected, reduced)
    self.assertAllEqual([5, 5, 4], length)

  def testJoinReducer(self):
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer()([1, 2, 3]))
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer()([(1,), (2,), (3,)]))
    self.assertTupleEqual((1, 2, 3), reducer.JoinReducer()([1, (2, 3)]))

    # Named tuples should not be unpacked.
    State = collections.namedtuple("State", ["h", "c"])
    self.assertTupleEqual((State(h=1, c=2), State(h=3, c=4), State(h=5, c=6)),
                          reducer.JoinReducer()([
                              State(h=1, c=2), (State(h=3, c=4), State(h=5, c=6))]))

  @parameterized.expand([
      [reducer.SumReducer(), [1, 2, 3], [4, 5, 6], [5, 7, 9]],
      [reducer.SumReducer(), (1, 2, 3), (4, 5, 6), (5, 7, 9)],
      [reducer.SumReducer(), 1, 2, 3]
  ])
  def testZipAndReduce(self, reducer, x, y, expected_z):
    z = reducer.zip_and_reduce(x, y)
    z = self.evaluate(z)
    self.assertAllEqual(z, expected_z)


if __name__ == "__main__":
  tf.test.main()
