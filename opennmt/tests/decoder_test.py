import math
import os

import tensorflow as tf
import numpy as np

from opennmt import decoders
from opennmt.decoders import decoder, self_attention_decoder
from opennmt.utils import beam_search
from opennmt.layers import bridge
from opennmt.tests import test_util


def _generate_source_context(batch_size,
                             depth,
                             initial_state_fn=None,
                             num_sources=1,
                             dtype=tf.float32):
    memory_sequence_length = [
        np.random.randint(1, high=20, size=batch_size) for _ in range(num_sources)]
    memory_time = [np.amax(length) for length in memory_sequence_length]
    memory = [
        tf.placeholder_with_default(
            np.random.randn(batch_size, time, depth).astype(dtype.as_numpy_dtype()),
            shape=(None, None, depth))
        for time in memory_time]
    if initial_state_fn is not None:
      initial_state = initial_state_fn(tf.shape(memory[0])[0], dtype)
    else:
      initial_state = None
    if num_sources == 1:
      memory_sequence_length = memory_sequence_length[0]
      memory = memory[0]
    return initial_state, memory, memory_sequence_length


class DecoderTest(tf.test.TestCase):

  def testSamplingProbability(self):
    step = tf.constant(5, dtype=tf.int64)
    large_step = tf.constant(1000, dtype=tf.int64)
    self.assertIsNone(decoder.get_sampling_probability(step))
    with self.assertRaises(ValueError):
      decoder.get_sampling_probability(step, schedule_type="linear")
    with self.assertRaises(ValueError):
      decoder.get_sampling_probability(step, schedule_type="linear", k=1)
    with self.assertRaises(TypeError):
      decoder.get_sampling_probability(step, schedule_type="foo", k=1)

    constant_sample_prob = decoder.get_sampling_probability(
        step, read_probability=0.9)
    linear_sample_prob = decoder.get_sampling_probability(
        step, read_probability=1.0, schedule_type="linear", k=0.1)
    linear_sample_prob_same = decoder.get_sampling_probability(
        step, read_probability=2.0, schedule_type="linear", k=0.1)
    linear_sample_prob_inf = decoder.get_sampling_probability(
        large_step, read_probability=1.0, schedule_type="linear", k=0.1)
    exp_sample_prob = decoder.get_sampling_probability(
        step, schedule_type="exponential", k=0.8)
    inv_sig_sample_prob = decoder.get_sampling_probability(
        step, schedule_type="inverse_sigmoid", k=1)

    self.assertAlmostEqual(0.1, constant_sample_prob)
    self.assertAlmostEqual(0.5, self.evaluate(linear_sample_prob))
    self.assertAlmostEqual(0.5, self.evaluate(linear_sample_prob_same))
    self.assertAlmostEqual(1.0, self.evaluate(linear_sample_prob_inf))
    self.assertAlmostEqual(1.0 - pow(0.8, 5), self.evaluate(exp_sample_prob))
    self.assertAlmostEqual(
        1.0 - (1.0 / (1.0 + math.exp(5.0 / 1.0))), self.evaluate(inv_sig_sample_prob))

  def _testDecoderTraining(self, decoder, initial_state_fn=None, num_sources=1, dtype=tf.float32):
    batch_size = 4
    vocab_size = 10
    time_dim = 5
    depth = 6
    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, time_dim, depth).astype(dtype.as_numpy_dtype()),
        shape=(None, None, depth))
    # NOTE: max(sequence_length) may be less than time_dim when num_gpus > 1
    sequence_length = [1, 3, 4, 2]
    initial_state, memory, memory_sequence_length = _generate_source_context(
        batch_size,
        depth,
        initial_state_fn=initial_state_fn,
        num_sources=num_sources,
        dtype=dtype)
    outputs, _, _, attention = decoder.decode(
        inputs,
        sequence_length,
        vocab_size=vocab_size,
        initial_state=initial_state,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        return_alignment_history=True)
    self.assertEqual(outputs.dtype, dtype)
    output_time_dim = tf.shape(outputs)[1]
    if decoder.support_alignment_history and num_sources == 1:
      self.assertIsNotNone(attention)
    else:
      self.assertIsNone(attention)

    saver = tf.train.Saver(var_list=tf.global_variables())
    with self.test_session(graph=tf.get_default_graph()) as sess:
      sess.run(tf.global_variables_initializer())
      output_time_dim_val = sess.run(output_time_dim)
      self.assertEqual(time_dim, output_time_dim_val)
      if decoder.support_alignment_history and num_sources == 1:
        attention_val, memory_time = sess.run([attention, tf.shape(memory)[1]])
        self.assertAllEqual([batch_size, time_dim, memory_time], attention_val.shape)
      return saver.save(sess, os.path.join(self.get_temp_dir(), "model.ckpt"))

  def _testDecoderInference(self,
                            decoder,
                            initial_state_fn=None,
                            num_sources=1,
                            with_beam_search=False,
                            with_alignment_history=False,
                            dtype=tf.float32,
                            checkpoint_path=None):
    batch_size = 4
    beam_width = 5
    num_hyps = beam_width if with_beam_search else 1
    vocab_size = 10
    depth = 6
    end_token = 2
    start_tokens = tf.placeholder_with_default([1] * batch_size, shape=[None])
    embedding =  tf.placeholder_with_default(
        np.random.randn(vocab_size, depth).astype(dtype.as_numpy_dtype()),
        shape=(vocab_size, depth))
    initial_state, memory, memory_sequence_length = _generate_source_context(
        batch_size,
        depth,
        initial_state_fn=initial_state_fn,
        num_sources=num_sources,
        dtype=dtype)

    if with_beam_search:
      decode_fn = decoder.dynamic_decode_and_search
    else:
      decode_fn = decoder.dynamic_decode

    additional_kwargs = {}
    if with_alignment_history:
      additional_kwargs["return_alignment_history"] = True
    if with_beam_search:
      additional_kwargs["beam_width"] = beam_width

    outputs = decode_fn(
        embedding,
        start_tokens,
        end_token,
        vocab_size=vocab_size,
        initial_state=initial_state,
        maximum_iterations=10,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        **additional_kwargs)

    ids = outputs[0]
    state = outputs[1]
    lengths = outputs[2]
    log_probs = outputs[3]
    self.assertEqual(log_probs.dtype, tf.float32)

    saver = tf.train.Saver(var_list=tf.global_variables())

    with self.test_session(graph=tf.get_default_graph()) as sess:
      if checkpoint_path is not None:
        saver.restore(sess, checkpoint_path)
      else:
        sess.run(tf.global_variables_initializer())

      if not with_alignment_history:
        self.assertEqual(4, len(outputs))
      else:
        self.assertEqual(5, len(outputs))
        alignment_history = outputs[4]
        if decoder.support_alignment_history and num_sources == 1:
          self.assertIsInstance(alignment_history, tf.Tensor)
          alignment_history, decode_time, memory_time = sess.run(
              [alignment_history, tf.shape(ids)[-1], tf.shape(memory)[1]])
          self.assertAllEqual(
              [batch_size, num_hyps, decode_time - 1, memory_time], alignment_history.shape)
        else:
          self.assertIsNone(alignment_history)

      ids, lengths, log_probs = sess.run([ids, lengths, log_probs])
      self.assertAllEqual([batch_size, num_hyps], ids.shape[0:2])
      self.assertAllEqual([batch_size, num_hyps], lengths.shape)
      self.assertAllEqual([batch_size, num_hyps], log_probs.shape)

  def _testDecoder(self, decoder, initial_state_fn=None, num_sources=1, dtype=tf.float32):
    with tf.Graph().as_default() as g:
      checkpoint_path = self._testDecoderTraining(
          decoder,
          initial_state_fn=initial_state_fn,
          num_sources=num_sources,
          dtype=dtype)

    with tf.Graph().as_default() as g:
      self._testDecoderInference(
          decoder,
          initial_state_fn=initial_state_fn,
          num_sources=num_sources,
          with_beam_search=False,
          with_alignment_history=False,
          dtype=dtype,
          checkpoint_path=checkpoint_path)
    with tf.Graph().as_default() as g:
      self._testDecoderInference(
          decoder,
          initial_state_fn=initial_state_fn,
          num_sources=num_sources,
          with_beam_search=False,
          with_alignment_history=True,
          dtype=dtype,
          checkpoint_path=checkpoint_path)
    with tf.Graph().as_default() as g:
      self._testDecoderInference(
          decoder,
          initial_state_fn=initial_state_fn,
          num_sources=num_sources,
          with_beam_search=True,
          with_alignment_history=False,
          dtype=dtype,
          checkpoint_path=checkpoint_path)
    with tf.Graph().as_default() as g:
      self._testDecoderInference(
          decoder,
          initial_state_fn=initial_state_fn,
          num_sources=num_sources,
          with_beam_search=True,
          with_alignment_history=True,
          dtype=dtype,
          checkpoint_path=checkpoint_path)

  @test_util.run_tf1_only
  def testRNNDecoder(self):
    decoder = decoders.RNNDecoder(2, 20)
    self._testDecoder(decoder)

  @test_util.run_tf1_only
  def testAttentionalRNNDecoder(self):
    decoder = decoders.AttentionalRNNDecoder(2, 20)
    self._testDecoder(decoder)

  @test_util.run_tf1_only
  def testAttentionalRNNDecoderWithDenseBridge(self):
    decoder = decoders.AttentionalRNNDecoder(2, 36, bridge=bridge.DenseBridge())
    encoder_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(5),
                                                tf.nn.rnn_cell.LSTMCell(5)])
    initial_state_fn = lambda batch_size, dtype: encoder_cell.zero_state(batch_size, dtype)
    self._testDecoder(decoder, initial_state_fn=initial_state_fn)

  @test_util.run_tf1_only
  def testMultiAttentionalRNNDecoder(self):
    decoder = decoders.MultiAttentionalRNNDecoder(2, 20, attention_layers=[0])
    self._testDecoder(decoder)

  @test_util.run_tf1_only
  def testRNMTPlusDecoder(self):
    decoder = decoders.RNMTPlusDecoder(2, 20, 4)
    self._testDecoder(decoder)

  @test_util.run_tf1_only
  def testSelfAttentionDecoder(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoder(decoder)

  @test_util.run_tf1_only
  def testSelfAttentionDecoderFP16(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoder(decoder, dtype=tf.float16)

  @test_util.run_tf1_only
  def testSelfAttentionDecoderMultiSource(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoder(decoder, num_sources=2)

  def testPenalizeToken(self):
    log_probs = tf.zeros([4, 6])
    token_id = 1
    log_probs = beam_search.penalize_token(log_probs, token_id)
    log_probs = self.evaluate(log_probs)
    self.assertTrue(np.all(log_probs[:, token_id] < 0))
    non_penalized = np.delete(log_probs, 1, token_id)
    self.assertEqual(np.sum(non_penalized), 0)

  @test_util.run_tf2_only
  def testSelfAttentionDecoderV2(self):
    decoder = self_attention_decoder.SelfAttentionDecoderV2(
        2, num_units=6, num_heads=2, ffn_inner_dim=12)
    decoder.initialize(vocab_size=10)
    inputs = tf.random.uniform([3, 5, 6])
    sequence_length = tf.constant([5, 5, 4], dtype=tf.int32)
    memory = tf.random.uniform([3, 7, 6])
    memory_sequence_length = tf.constant([5, 7, 3], dtype=tf.int32)
    logits, state, attention = decoder(
        inputs,
        sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=True)
    self.assertEqual(logits.shape[-1], 10)
    self.assertEqual(len(state), 2)
    self.assertListEqual(attention.shape.as_list(), [3, 5, 7])
    state = decoder.get_initial_state(batch_size=3)
    inputs = tf.random.uniform([3, 6])
    logits, state, attention = decoder(
        inputs,
        tf.constant(0),
        state=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)
    self.assertEqual(logits.shape[-1], 10)
    self.assertEqual(len(state), 2)
    self.assertListEqual(attention.shape.as_list(), [3, 7])

  @test_util.run_tf2_only
  def testSelfAttentionDecoderV2MultiSource(self):
    decoder = self_attention_decoder.SelfAttentionDecoderV2(
        2, num_units=6, num_heads=2, ffn_inner_dim=12, num_sources=2)
    decoder.initialize(vocab_size=10)
    inputs = tf.random.uniform([3, 5, 6])
    sequence_length = tf.constant([5, 5, 4], dtype=tf.int32)
    memory = [tf.random.uniform([3, 7, 6]), tf.random.uniform([3, 2, 6])]
    memory_sequence_length = [
          tf.constant([5, 7, 3], dtype=tf.int32), tf.constant([1, 1, 2], dtype=tf.int32)]
    logits, state, attention = decoder(
        inputs,
        sequence_length,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        training=True)
    self.assertEqual(logits.shape[-1], 10)
    self.assertIsNone(attention)
    state = decoder.get_initial_state(batch_size=3)
    inputs = tf.random.uniform([3, 6])
    logits, state, attention = decoder(
        inputs,
        tf.constant(0),
        state=state,
        memory=memory,
        memory_sequence_length=memory_sequence_length)
    self.assertEqual(logits.shape[-1], 10)
    self.assertEqual(len(state), 2)
    self.assertEqual(len(state[0]["memory_kv"]), 2)
    self.assertIsNone(attention)


if __name__ == "__main__":
  tf.test.main()
