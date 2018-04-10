import tensorflow as tf
import numpy as np

from opennmt import encoders
from opennmt.layers import reducer


def _build_dummy_sequences(sequence_length, depth=5):
  batch_size = len(sequence_length)
  return tf.placeholder_with_default(
      np.random.randn(
          batch_size, max(sequence_length), depth).astype(np.float32),
      shape=(None, None, depth))


class EncoderTest(tf.test.TestCase):

  def testSelfAttentionEncoder(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length, depth=10)
    encoder = encoders.SelfAttentionEncoder(
        3, num_units=36, num_heads=4, ffn_inner_dim=52)
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=tf.constant(sequence_length))
    self.assertEqual(3, len(state))
    for s in state:
      self.assertIsInstance(s, tf.Tensor)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 21, 36], outputs.shape)
      self.assertAllEqual(sequence_length, encoded_length)

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
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(3, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
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
    _, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(4, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoded_length = sess.run(encoded_length)
      self.assertAllEqual([4, 5, 5], encoded_length)

  def _testGoogleRNNEncoder(self, num_layers):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.GoogleRNNEncoder(num_layers, 10)
    outputs, state, _ = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(num_layers, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertAllEqual([3, max(sequence_length), 10], outputs.shape)

  def testGoogleRNNEncoder2Layers(self):
    self._testGoogleRNNEncoder(2)
  def testGoogleRNNEncoder3Layers(self):
    self._testGoogleRNNEncoder(3)

  def testParallelEncoder(self):
    sequence_lengths = [[17, 21, 20], [10, 9, 15]]
    inputs = [
        _build_dummy_sequences(length) for length in sequence_lengths]
    encoder = encoders.ParallelEncoder([
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.UnidirectionalRNNEncoder(1, 20)],
        outputs_reducer=reducer.ConcatReducer(axis=1))
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_lengths)
    self.assertEqual(2, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
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
