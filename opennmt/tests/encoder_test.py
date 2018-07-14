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

  def testPyramidalEncoderShortSequences(self):
    sequence_length = [3, 4, 2]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.PyramidalRNNEncoder(3, 10, reduction_factor=2)
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoded_length = sess.run(encoded_length)
      self.assertAllEqual([1, 1, 1], encoded_length)

  def _testSequentialEncoder(self, transition_layer_fn=None):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoders_sequence = [
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.PyramidalRNNEncoder(3, 10, reduction_factor=2)]
    encoder = encoders.SequentialEncoder(
        encoders_sequence, transition_layer_fn=transition_layer_fn)
    _, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(4, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      encoded_length = sess.run(encoded_length)
      self.assertAllEqual([4, 5, 5], encoded_length)

  def testSequentialEncoder(self):
    self._testSequentialEncoder()

  def testSequentialEncoderWithTransitionLayer(self):
    layer_norm_fn = lambda x: tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
    self._testSequentialEncoder(transition_layer_fn=layer_norm_fn)

  def testSequentialEncoderWithTransitionLayerList(self):
    layer_norm_fn = lambda x: tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
    self._testSequentialEncoder(transition_layer_fn=[layer_norm_fn])

  def testSequentialEncoderWithInvalidTransitionLayerList(self):
    layer_norm_fn = lambda x: tf.contrib.layers.layer_norm(x, begin_norm_axis=-1)
    with self.assertRaises(ValueError):
      self._testSequentialEncoder(transition_layer_fn=[layer_norm_fn, layer_norm_fn])

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

  def testRNMTPlusEncoder(self):
    sequence_length = [4, 6, 5]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.RNMTPlusEncoder(6, 10)
    outputs, state, _ = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(6, len(state))
    for s in state:
      self.assertIsInstance(s, tf.contrib.rnn.LSTMStateTuple)
    self.assertEqual(10 * 2, state[0].h.get_shape().as_list()[-1])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertAllEqual([3, max(sequence_length), 10], outputs.shape)

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

  def _encodeInParallel(self,
                        inputs,
                        sequence_length=None,
                        outputs_layer_fn=None,
                        combined_output_layer_fn=None):
    columns = [
        encoders.UnidirectionalRNNEncoder(1, 20),
        encoders.UnidirectionalRNNEncoder(1, 20)]
    encoder = encoders.ParallelEncoder(
        columns,
        outputs_reducer=reducer.ConcatReducer(),
        outputs_layer_fn=outputs_layer_fn,
        combined_output_layer_fn=combined_output_layer_fn)
    return encoder.encode(inputs, sequence_length=sequence_length)

  def testParallelEncoderSameInput(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    outputs, _, encoded_length = self._encodeInParallel(
        inputs, sequence_length=sequence_length)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 21, 40], outputs.shape)
      self.assertAllEqual(sequence_length, encoded_length)

  def testParallelEncoderCombinedOutputLayer(self):
    sequence_length = [4, 6, 5]
    inputs = _build_dummy_sequences(sequence_length)
    outputs, _, encoded_length = self._encodeInParallel(
        inputs,
        sequence_length=sequence_length,
        combined_output_layer_fn=lambda x: tf.layers.dense(x, 15))
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertEqual(15, outputs.shape[-1])

  def _encodeAndProjectInParallel(self, outputs_size):
    sequence_length = [4, 6, 5]
    inputs = _build_dummy_sequences(sequence_length)
    if isinstance(outputs_size, list):
      outputs_layer_fn = [lambda x, s=s: tf.layers.dense(x, s) for s in outputs_size]
      combined_output_size = sum(outputs_size)
    else:
      outputs_layer_fn = lambda x: tf.layers.dense(x, outputs_size)
      combined_output_size = outputs_size * 2
    outputs, _, encoded_length = self._encodeInParallel(
        inputs,
        sequence_length=sequence_length,
        outputs_layer_fn=outputs_layer_fn)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertEqual(combined_output_size, outputs.shape[-1])

  def testParallelEncoderSameOutputsLayer(self):
    self._encodeAndProjectInParallel(15)

  def testParallelEncoderOutputsLayer(self):
    self._encodeAndProjectInParallel([14, 15])

  def testParallelEncoderOutputsLayerInvalid(self):
    with self.assertRaises(ValueError):
      self._encodeAndProjectInParallel([15])


if __name__ == "__main__":
  tf.test.main()
