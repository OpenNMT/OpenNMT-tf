import tensorflow as tf
import numpy as np

from opennmt import encoders
from opennmt.utils import reducer


def _build_dummy_sequences(sequence_length):
  batch_size = len(sequence_length)
  depth = 5
  return tf.placeholder_with_default(
      np.random.randn(
          batch_size, max(sequence_length), depth).astype(np.float32),
      shape=(None, None, depth))


class EncoderTest(tf.test.TestCase):

  def testConvEncoder(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.ConvEncoder(3, 10)
    outputs, _, encoded_length = encoder.encode(
        inputs, sequence_length=tf.constant(sequence_length))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 21, 10], outputs.shape)
      self.assertAllEqual(sequence_length, encoded_length)

  def testPyramidalEncoder(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.PyramidalRNNEncoder(3, 10, reduction_factor=2)
    outputs, _, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 6, 10], outputs.shape)
      self.assertAllEqual([4, 5, 5], encoded_length)

  def testSequentialEncoder(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.SequentialEncoder([
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.PyramidalRNNEncoder(3, 10, reduction_factor=2)])
    _, _, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoded_length = sess.run(encoded_length)
      self.assertAllEqual([4, 5, 5], encoded_length)

  def testParallelEncoder(self):
    sequence_lengths = [[17, 21, 20], [10, 9, 15]]
    inputs = [
        _build_dummy_sequences(length) for length in sequence_lengths]
    encoder = encoders.ParallelEncoder([
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.UnidirectionalRNNEncoder(1, 20)],
        outputs_reducer=reducer.ConcatReducer(axis=1))
    outputs, _, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_lengths)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 35, 20], outputs.shape)
      self.assertAllEqual([27, 30, 35], encoded_length)

  def testParallelEncoderSameInput(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.ParallelEncoder([
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.UnidirectionalRNNEncoder(1, 20)],
        outputs_reducer=reducer.ConcatReducer())
    outputs, _, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 21, 40], outputs.shape)
      self.assertAllEqual(sequence_length, encoded_length)


if __name__ == "__main__":
  tf.test.main()
