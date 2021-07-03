import math

import tensorflow as tf

from opennmt import decoders
from opennmt.layers import bridge


def _generate_source_context(batch_size, depth, num_sources=1, dtype=tf.float32):
    memory_sequence_length = [
        tf.random.uniform([batch_size], minval=1, maxval=20, dtype=tf.int32)
        for _ in range(num_sources)
    ]
    memory_time = [tf.reduce_max(length) for length in memory_sequence_length]
    memory = [
        tf.random.uniform([batch_size, time, depth], dtype=dtype)
        for time in memory_time
    ]
    initial_state = tuple(None for _ in range(num_sources))
    if num_sources == 1:
        memory_sequence_length = memory_sequence_length[0]
        memory = memory[0]
        initial_state = initial_state[0]
    return memory, memory_sequence_length, initial_state


class DecoderTest(tf.test.TestCase):
    def testSamplingProbability(self):
        step = tf.constant(5, dtype=tf.int64)
        large_step = tf.constant(1000, dtype=tf.int64)
        self.assertIsNone(decoders.get_sampling_probability(step))
        with self.assertRaises(ValueError):
            decoders.get_sampling_probability(step, schedule_type="linear")
        with self.assertRaises(ValueError):
            decoders.get_sampling_probability(step, schedule_type="linear", k=1)
        with self.assertRaises(TypeError):
            decoders.get_sampling_probability(step, schedule_type="foo", k=1)

        constant_sample_prob = decoders.get_sampling_probability(
            step, read_probability=0.9
        )
        linear_sample_prob = decoders.get_sampling_probability(
            step, read_probability=1.0, schedule_type="linear", k=0.1
        )
        linear_sample_prob_same = decoders.get_sampling_probability(
            step, read_probability=2.0, schedule_type="linear", k=0.1
        )
        linear_sample_prob_inf = decoders.get_sampling_probability(
            large_step, read_probability=1.0, schedule_type="linear", k=0.1
        )
        exp_sample_prob = decoders.get_sampling_probability(
            step, schedule_type="exponential", k=0.8
        )
        inv_sig_sample_prob = decoders.get_sampling_probability(
            step, schedule_type="inverse_sigmoid", k=1
        )

        self.assertAlmostEqual(0.1, constant_sample_prob)
        self.assertAlmostEqual(0.5, self.evaluate(linear_sample_prob))
        self.assertAlmostEqual(0.5, self.evaluate(linear_sample_prob_same))
        self.assertAlmostEqual(1.0, self.evaluate(linear_sample_prob_inf))
        self.assertAlmostEqual(1.0 - pow(0.8, 5), self.evaluate(exp_sample_prob))
        self.assertAlmostEqual(
            1.0 - (1.0 / (1.0 + math.exp(5.0 / 1.0))),
            self.evaluate(inv_sig_sample_prob),
        )

    def _testDecoder(
        self, decoder, initial_state_fn=None, num_sources=1, dtype=tf.float32
    ):
        batch_size = 4
        vocab_size = 10
        time_dim = 5
        depth = 6
        memory, memory_sequence_length, initial_state = _generate_source_context(
            batch_size, depth, num_sources=num_sources, dtype=dtype
        )

        if initial_state_fn is not None:
            initial_state = initial_state_fn(batch_size, dtype)
        decoder.initialize(vocab_size=vocab_size)
        initial_state = decoder.initial_state(
            memory=memory,
            memory_sequence_length=memory_sequence_length,
            initial_state=initial_state,
            dtype=dtype,
        )

        # Test 3D inputs.
        inputs = tf.random.uniform([batch_size, time_dim, depth], dtype=dtype)
        # Allow max(sequence_length) to be less than time_dim.
        sequence_length = tf.constant([1, 3, 4, 2], dtype=tf.int32)
        outputs, _, attention = decoder(
            inputs, sequence_length, state=initial_state, training=True
        )
        self.assertEqual(outputs.dtype, dtype)
        output_time_dim = tf.shape(outputs)[1]
        if decoder.support_alignment_history:
            self.assertIsNotNone(attention)
        else:
            self.assertIsNone(attention)
        output_time_dim_val = self.evaluate(output_time_dim)
        self.assertEqual(time_dim, output_time_dim_val)
        if decoder.support_alignment_history:
            first_memory = memory[0] if isinstance(memory, list) else memory
            attention_val, memory_time = self.evaluate(
                [attention, tf.shape(first_memory)[1]]
            )
            self.assertAllEqual(
                [batch_size, time_dim, memory_time], attention_val.shape
            )

        # Test 2D inputs.
        inputs = tf.random.uniform([batch_size, depth], dtype=dtype)
        step = tf.constant(0, dtype=tf.int32)
        outputs, _, attention = decoder(inputs, step, state=initial_state)
        self.assertEqual(outputs.dtype, dtype)
        if decoder.support_alignment_history:
            self.assertIsNotNone(attention)
        else:
            self.assertIsNone(attention)
        self.evaluate(outputs)

    def testRNNDecoder(self):
        decoder = decoders.RNNDecoder(2, 20)
        self._testDecoder(decoder)

    def testAttentionalRNNDecoder(self):
        decoder = decoders.AttentionalRNNDecoder(2, 20)
        self._testDecoder(decoder)

    def testAttentionalRNNDecoderWithDenseBridge(self):
        decoder = decoders.AttentionalRNNDecoder(2, 36, bridge_class=bridge.DenseBridge)
        encoder_cell = tf.keras.layers.StackedRNNCells(
            [tf.keras.layers.LSTMCell(5), tf.keras.layers.LSTMCell(5)]
        )
        initial_state_fn = lambda batch_size, dtype: encoder_cell.get_initial_state(
            batch_size=batch_size, dtype=dtype
        )
        self._testDecoder(decoder, initial_state_fn=initial_state_fn)

    def testAttentionalRNNDecoderFirstLayer(self):
        decoder = decoders.AttentionalRNNDecoder(2, 20, first_layer_attention=True)
        self._testDecoder(decoder)

    def testRNMTPlusDecoder(self):
        decoder = decoders.RNMTPlusDecoder(2, 20, 4)
        self._testDecoder(decoder)

    def testSelfAttentionDecoder(self):
        decoder = decoders.SelfAttentionDecoder(
            num_layers=2,
            num_units=6,
            num_heads=2,
            ffn_inner_dim=12,
            vocab_size=10,
        )
        self.assertTrue(decoder.initialized)
        self._testDecoder(decoder)

    def testSelfAttentionDecoderMultiSource(self):
        num_sources = 2
        decoder = decoders.SelfAttentionDecoder(
            2, num_units=6, num_heads=2, ffn_inner_dim=12, num_sources=num_sources
        )
        self._testDecoder(decoder, num_sources=num_sources)

    def testSelfAttentionDecoderWithoutSourceLength(self):
        batch_size = 4
        depth = 6
        decoder = decoders.SelfAttentionDecoder(
            num_layers=2,
            num_units=depth,
            num_heads=2,
            ffn_inner_dim=depth * 2,
            vocab_size=10,
        )

        memory, _, _ = _generate_source_context(batch_size, depth)
        inputs = tf.random.uniform([batch_size, depth])
        step = tf.constant(0)
        initial_state = decoder.initial_state(memory)
        decoder(inputs, step, state=initial_state)


if __name__ == "__main__":
    tf.test.main()
