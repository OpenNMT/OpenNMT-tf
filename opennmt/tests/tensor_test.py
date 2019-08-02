import tensorflow as tf

from opennmt.utils import tensor as tensor_util


class TensorTest(tf.test.TestCase):

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

    rolled = tensor_util.roll_sequence(tensor, offset)
    self.assertAllEqual(expected, self.evaluate(rolled))


if __name__ == "__main__":
  tf.test.main()
