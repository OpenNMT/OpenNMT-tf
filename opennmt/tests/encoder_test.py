from parameterized import parameterized

import tensorflow as tf
import numpy as np

from opennmt import encoders
from opennmt.encoders import rnn_encoder, self_attention_encoder
from opennmt.layers import reducer
from opennmt.utils import compat
from opennmt.tests import test_util


def _build_dummy_sequences(sequence_length, depth=5, dtype=tf.float32):
  batch_size = len(sequence_length)
  return tf.placeholder_with_default(
      np.random.randn(
          batch_size, max(sequence_length), depth).astype(dtype.as_numpy_dtype()),
      shape=(None, None, depth))


class DenseEncoder(encoders.Encoder):

  def __init__(self, num_layers, num_units):
    super(DenseEncoder, self).__init__()
    self.layers = [tf.keras.layers.Dense(num_units) for _ in range(num_layers)]

  def call(self, inputs, sequence_length=None, training=None):
    states = []
    for layer in self.layers:
      inputs = layer(inputs)
      states.append(inputs[:, -1])
    return inputs, tuple(states), sequence_length


class EncoderTest(tf.test.TestCase):

  def _testSelfAttentionEncoder(self, dtype=tf.float32):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length, depth=10, dtype=dtype)
    encoder = encoders.SelfAttentionEncoder(
        3, num_units=36, num_heads=4, ffn_inner_dim=52)
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=tf.constant(sequence_length))
    self.assertEqual(outputs.dtype, dtype)
    self.assertEqual(3, len(state))
    for s in state:
      self.assertIsInstance(s, tf.Tensor)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 21, 36], outputs.shape)
      self.assertAllEqual(sequence_length, encoded_length)

  @test_util.skip_if_unsupported("RaggedTensor")
  def testMeanEncoder(self):
    inputs = tf.concat([tf.ones([1, 5, 1]), 2*tf.ones([1, 5, 1])], 0)
    length = tf.constant([2, 4], dtype=tf.int32)
    mask = tf.sequence_mask(length, maxlen=tf.shape(inputs)[1], dtype=inputs.dtype)
    inputs *= tf.expand_dims(mask, -1)
    encoder = encoders.MeanEncoder()
    _, state, _ = encoder.encode(inputs, sequence_length=length)
    state = self.evaluate(state)
    self.assertEqual(state[0][0], 1)
    self.assertEqual(state[1][0], 2)

  @test_util.run_tf1_only
  def testSelfAttentionEncoder(self):
    self._testSelfAttentionEncoder(dtype=tf.float32)

  @test_util.run_tf1_only
  def testSelfAttentionEncoderFP16(self):
    self._testSelfAttentionEncoder(dtype=tf.float16)

  @test_util.run_tf1_only
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

  @test_util.run_tf1_only
  def testPyramidalEncoder(self):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.PyramidalRNNEncoder(3, 10, reduction_factor=2)
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(3, len(state))
    for s in state:
      self.assertIsInstance(s, tf.nn.rnn_cell.LSTMStateTuple)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs, encoded_length = sess.run([outputs, encoded_length])
      self.assertAllEqual([3, 6, 10], outputs.shape)
      self.assertAllEqual([4, 5, 5], encoded_length)

  @test_util.run_tf1_only
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

  @parameterized.expand([[None], [tf.identity], [[tf.identity]]])
  def testSequentialEncoder(self, transition_layer_fn):
    inputs = tf.zeros([3, 5, 10])
    encoder = encoders.SequentialEncoder(
        [DenseEncoder(1, 20), DenseEncoder(3, 20)],
        transition_layer_fn=transition_layer_fn)
    outputs, states, _ = encoder.encode(inputs)
    self.assertEqual(len(states), 4)
    if not compat.is_tf2():
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
    outputs = self.evaluate(outputs)
    self.assertAllEqual(outputs.shape, [3, 5, 20])

  def testSequentialEncoderWithTooManyTransitionLayers(self):
    with self.assertRaises(ValueError):
      _ = encoders.SequentialEncoder(
          [DenseEncoder(1, 20), DenseEncoder(3, 20)],
          transition_layer_fn=[tf.identity, tf.identity])

  def _testGoogleRNNEncoder(self, num_layers):
    sequence_length = [17, 21, 20]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.GoogleRNNEncoder(num_layers, 10)
    outputs, state, _ = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(num_layers, len(state))
    for s in state:
      self.assertIsInstance(s, tf.nn.rnn_cell.LSTMStateTuple)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertAllEqual([3, max(sequence_length), 10], outputs.shape)

  @test_util.run_tf1_only
  def testGoogleRNNEncoder2Layers(self):
    self._testGoogleRNNEncoder(2)
  @test_util.run_tf1_only
  def testGoogleRNNEncoder3Layers(self):
    self._testGoogleRNNEncoder(3)

  @test_util.run_tf1_only
  def testRNMTPlusEncoder(self):
    sequence_length = [4, 6, 5]
    inputs = _build_dummy_sequences(sequence_length)
    encoder = encoders.RNMTPlusEncoder(6, 10)
    outputs, state, _ = encoder.encode(
        inputs, sequence_length=sequence_length)
    self.assertEqual(6, len(state))
    for s in state:
      self.assertIsInstance(s, tf.nn.rnn_cell.LSTMStateTuple)
    self.assertEqual(10 * 2, state[0].h.get_shape().as_list()[-1])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      outputs = sess.run(outputs)
      self.assertAllEqual([3, max(sequence_length), 10], outputs.shape)

  def testParallelEncoder(self):
    sequence_lengths = [[3, 5, 2], [6, 6, 4]]
    inputs = [tf.zeros([3, 5, 10]), tf.zeros([3, 6, 10])]
    encoder = encoders.ParallelEncoder(
        [DenseEncoder(1, 20), DenseEncoder(2, 20)],
        outputs_reducer=reducer.ConcatReducer(axis=1))
    outputs, state, encoded_length = encoder.encode(
        inputs, sequence_length=sequence_lengths)
    self.assertEqual(len(state), 3)
    if not compat.is_tf2():
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
    outputs, encoded_length = self.evaluate([outputs, encoded_length])
    self.assertAllEqual([3, 11, 20], outputs.shape)
    self.assertAllEqual([9, 11, 6], encoded_length)

  def _encodeInParallel(self,
                        inputs,
                        sequence_length=None,
                        outputs_layer_fn=None,
                        combined_output_layer_fn=None):
    columns = [DenseEncoder(1, 20), DenseEncoder(1, 20)]
    encoder = encoders.ParallelEncoder(
        columns,
        outputs_reducer=reducer.ConcatReducer(),
        outputs_layer_fn=outputs_layer_fn,
        combined_output_layer_fn=combined_output_layer_fn)
    outputs, _, _ = encoder.encode(inputs, sequence_length=sequence_length)
    if not compat.is_tf2():
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
    return self.evaluate(outputs)

  def testParallelEncoderSameInput(self):
    sequence_length = tf.constant([2, 5, 4], dtype=tf.int32)
    inputs = tf.zeros([3, 5, 10])
    outputs = self._encodeInParallel(inputs, sequence_length=sequence_length)
    self.assertAllEqual(outputs.shape, [3, 5, 40])

  def testParallelEncoderCombinedOutputLayer(self):
    sequence_length = tf.constant([2, 5, 4], dtype=tf.int32)
    inputs = tf.zeros([3, 5, 10])
    outputs = self._encodeInParallel(
        inputs,
        sequence_length=sequence_length,
        combined_output_layer_fn=tf.keras.layers.Dense(15))
    self.assertEqual(outputs.shape[-1], 15)

  def _encodeAndProjectInParallel(self, outputs_size):
    sequence_length = tf.constant([2, 5, 4], dtype=tf.int32)
    inputs = tf.zeros([3, 5, 10])
    if isinstance(outputs_size, list):
      outputs_layer_fn = [tf.keras.layers.Dense(s) for s in outputs_size]
      combined_output_size = sum(outputs_size)
    else:
      outputs_layer_fn = tf.keras.layers.Dense(outputs_size)
      combined_output_size = outputs_size * 2
    outputs = self._encodeInParallel(
        inputs,
        sequence_length=sequence_length,
        outputs_layer_fn=outputs_layer_fn)
    self.assertEqual(outputs.shape[-1], combined_output_size)

  def testParallelEncoderSameOutputsLayer(self):
    self._encodeAndProjectInParallel(15)

  def testParallelEncoderOutputsLayer(self):
    self._encodeAndProjectInParallel([14, 15])

  def testParallelEncoderOutputsLayerInvalid(self):
    with self.assertRaises(ValueError):
      self._encodeAndProjectInParallel([15])

  def testParallelEncoderReuse(self):
    lengths = [tf.constant([2, 5, 4], dtype=tf.int32), tf.constant([6, 6, 3], dtype=tf.int32)]
    inputs = [tf.zeros([3, 5, 10]), tf.zeros([3, 6, 10])]
    encoder = encoders.ParallelEncoder(DenseEncoder(2, 20), outputs_reducer=None)
    outputs, _, _ = encoder.encode(inputs, sequence_length=lengths)
    if not compat.is_tf2():
      with self.test_session() as sess:
        sess.run(tf.global_variables_initializer())
    outputs = self.evaluate(outputs)
    self.assertIsInstance(outputs, tuple)
    self.assertEqual(len(outputs), 2)

  @parameterized.expand([[tf.float32], [tf.float16]])
  @test_util.run_tf2_only
  def testSelfAttentionEncoderV2(self, dtype):
    encoder = self_attention_encoder.SelfAttentionEncoderV2(
        3, num_units=20, num_heads=4, ffn_inner_dim=40)
    inputs = tf.random.uniform([4, 5, 10], dtype=dtype)
    lengths = tf.constant([4, 3, 5, 2])
    outputs, _, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(outputs.dtype, dtype)

  @parameterized.expand([[tf.keras.layers.LSTMCell], [tf.keras.layers.GRUCell]])
  @test_util.run_tf2_only
  def testUnidirectionalRNNEncoderV2(self, cell_class):
    encoder = rnn_encoder.RNNEncoderV2(3, 20, cell_class=cell_class)
    inputs = tf.random.uniform([4, 5, 10])
    lengths = tf.constant([4, 3, 5, 2])
    outputs, states, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(len(states), 3)

  @parameterized.expand([[tf.keras.layers.LSTMCell], [tf.keras.layers.GRUCell]])
  @test_util.run_tf2_only
  def testBidirectionalRNNEncoderV2(self, cell_class):
    encoder = rnn_encoder.RNNEncoderV2(3, 20, bidirectional=True, cell_class=cell_class)
    inputs = tf.random.uniform([4, 5, 10])
    lengths = tf.constant([4, 3, 5, 2])
    outputs, states, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 40])
    self.assertEqual(len(states), 3)

  @test_util.run_tf2_only
  def testGNMTEncoder(self):
    encoder = rnn_encoder.GNMTEncoder(3, 20)
    inputs = tf.random.uniform([4, 5, 10])
    lengths = tf.constant([4, 3, 5, 2])
    outputs, states, _ = encoder(inputs, sequence_length=lengths, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertEqual(len(states), 3)


if __name__ == "__main__":
  tf.test.main()
