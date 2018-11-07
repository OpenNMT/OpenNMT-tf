import math

import tensorflow as tf
import numpy as np

from opennmt import decoders
from opennmt.decoders import decoder
from opennmt.utils import beam_search


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

    with self.test_session() as sess:
      self.assertAlmostEqual(0.1, constant_sample_prob)
      self.assertAlmostEqual(0.5, sess.run(linear_sample_prob))
      self.assertAlmostEqual(0.5, sess.run(linear_sample_prob_same))
      self.assertAlmostEqual(1.0, sess.run(linear_sample_prob_inf))
      self.assertAlmostEqual(1.0 - pow(0.8, 5), sess.run(exp_sample_prob))
      self.assertAlmostEqual(
          1.0 - (1.0 / (1.0 + math.exp(5.0 / 1.0))), sess.run(inv_sig_sample_prob))

  def _testDecoderTraining(self, decoder, dtype=tf.float32):
    batch_size = 4
    vocab_size = 10
    time_dim = 5
    depth = 6
    inputs = tf.placeholder_with_default(
        np.random.randn(batch_size, time_dim, depth).astype(dtype.as_numpy_dtype()),
        shape=(None, None, depth))
    # NOTE: max(sequence_length) may be less than time_dim when num_gpus > 1
    sequence_length = [1, 3, 4, 2]
    memory_sequence_length = [3, 7, 5, 4]
    memory_time = max(memory_sequence_length)
    memory = tf.placeholder_with_default(
        np.random.randn(batch_size, memory_time, depth).astype(dtype.as_numpy_dtype()),
        shape=(None, None, depth))
    outputs, _, _, attention = decoder.decode(
        inputs,
        sequence_length,
        vocab_size=vocab_size,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        return_alignment_history=True)
    self.assertEqual(outputs.dtype, dtype)
    output_time_dim = tf.shape(outputs)[1]
    if decoder.support_alignment_history:
      self.assertIsNotNone(attention)
    else:
      self.assertIsNone(attention)

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
    with self.test_session() as sess:
      output_time_dim_val = sess.run(output_time_dim)
      self.assertEqual(time_dim, output_time_dim_val)
      if decoder.support_alignment_history:
        attention_val = sess.run(attention)
        self.assertAllEqual([batch_size, time_dim, memory_time], attention_val.shape)

  def testRNNDecoderTraining(self):
    decoder = decoders.RNNDecoder(2, 20)
    self._testDecoderTraining(decoder)

  def testAttentionalRNNDecoderTraining(self):
    decoder = decoders.AttentionalRNNDecoder(2, 20)
    self._testDecoderTraining(decoder)

  def testMultiAttentionalRNNDecoderTraining(self):
    decoder = decoders.MultiAttentionalRNNDecoder(2, 20, attention_layers=[0])
    self._testDecoderTraining(decoder)

  def testSelfAttentionDecoderTraining(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoderTraining(decoder)

  def testSelfAttentionDecoderFP16Training(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoderTraining(decoder, dtype=tf.float16)

  def _testDecoderGeneric(self,
                          decoder,
                          with_beam_search=False,
                          with_alignment_history=False,
                          dtype=tf.float32):
    batch_size = 4
    beam_width = 5
    num_hyps = beam_width if with_beam_search else 1
    vocab_size = 10
    depth = 6
    end_token = 2
    start_tokens = tf.placeholder_with_default([1] * batch_size, shape=[None])
    memory_sequence_length = [3, 7, 5, 4]
    memory_time = max(memory_sequence_length)
    memory =  tf.placeholder_with_default(
        np.random.randn(batch_size, memory_time, depth).astype(dtype.as_numpy_dtype()),
        shape=(None, None, depth))
    memory_sequence_length = tf.placeholder_with_default(memory_sequence_length, shape=[None])
    embedding =  tf.placeholder_with_default(
        np.random.randn(vocab_size, depth).astype(dtype.as_numpy_dtype()),
        shape=(vocab_size, depth))

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
        maximum_iterations=10,
        memory=memory,
        memory_sequence_length=memory_sequence_length,
        **additional_kwargs)

    ids = outputs[0]
    state = outputs[1]
    lengths = outputs[2]
    log_probs = outputs[3]
    self.assertEqual(log_probs.dtype, tf.float32)

    decode_time = tf.shape(ids)[-1]

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())

    if not with_alignment_history:
      self.assertEqual(4, len(outputs))
    else:
      self.assertEqual(5, len(outputs))
      alignment_history = outputs[4]
      if decoder.support_alignment_history:
        self.assertIsInstance(alignment_history, tf.Tensor)
        with self.test_session() as sess:
          alignment_history, decode_time = sess.run([alignment_history, decode_time])
          self.assertAllEqual(
              [batch_size, num_hyps, decode_time, memory_time], alignment_history.shape)
      else:
        self.assertIsNone(alignment_history)

    with self.test_session() as sess:
      ids, lengths, log_probs = sess.run([ids, lengths, log_probs])
      self.assertAllEqual([batch_size, num_hyps], ids.shape[0:2])
      self.assertAllEqual([batch_size, num_hyps], lengths.shape)
      self.assertAllEqual([batch_size, num_hyps], log_probs.shape)

  def _testDecoder(self, decoder, dtype=tf.float32):
    with tf.variable_scope(tf.get_variable_scope()):
      self._testDecoderGeneric(
          decoder,
          with_beam_search=False,
          with_alignment_history=False,
          dtype=dtype)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      self._testDecoderGeneric(
          decoder,
          with_beam_search=False,
          with_alignment_history=True,
          dtype=dtype)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      self._testDecoderGeneric(
          decoder,
          with_beam_search=True,
          with_alignment_history=False,
          dtype=dtype)
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      self._testDecoderGeneric(
          decoder,
          with_beam_search=True,
          with_alignment_history=True,
          dtype=dtype)

  def testRNNDecoder(self):
    decoder = decoders.RNNDecoder(2, 20)
    self._testDecoder(decoder)

  def testAttentionalRNNDecoder(self):
    decoder = decoders.AttentionalRNNDecoder(2, 20)
    self._testDecoder(decoder)

  def testMultiAttentionalRNNDecoder(self):
    decoder = decoders.MultiAttentionalRNNDecoder(2, 20, attention_layers=[0])
    self._testDecoder(decoder)

  def testSelfAttentionDecoder(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoder(decoder)

  def testSelfAttentionDecoderFP16(self):
    decoder = decoders.SelfAttentionDecoder(2, num_units=6, num_heads=2, ffn_inner_dim=12)
    self._testDecoder(decoder, dtype=tf.float16)

  def testPenalizeToken(self):
    log_probs = tf.zeros([4, 6])
    token_id = 1
    log_probs = beam_search.penalize_token(log_probs, token_id)
    with self.test_session() as sess:
      log_probs = sess.run(log_probs)
      self.assertTrue(np.all(log_probs[:, token_id] < 0))
      non_penalized = np.delete(log_probs, 1, token_id)
      self.assertEqual(np.sum(non_penalized), 0)


if __name__ == "__main__":
  tf.test.main()
