import tensorflow as tf

from opennmt.layers import rnn


class RNNTest(tf.test.TestCase):
    def testRNNCell(self):
        cell = rnn.make_rnn_cell(3, 10, dropout=0.1, residual_connections=True)
        inputs = tf.random.uniform([4, 5])
        states = cell.get_initial_state(inputs=inputs)
        outputs, states = cell(inputs, states, training=True)
        self.assertEqual(len(states), 3)
        self.assertListEqual(outputs.shape.as_list(), [4, 10])

    def testRNN(self):
        cell = rnn.make_rnn_cell(3, 10, dropout=0.1, residual_connections=True)
        rnn_layer = rnn.RNN(cell)
        inputs = tf.random.uniform([4, 5, 5])
        outputs, states = rnn_layer(inputs, training=True)
        self.assertListEqual(outputs.shape.as_list(), [4, 5, 10])
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), 3)

    def testBRNN(self):
        cell = rnn.make_rnn_cell(3, 10, dropout=0.1, residual_connections=True)
        rnn_layer = rnn.RNN(cell, bidirectional=True)
        inputs = tf.random.uniform([4, 5, 5])
        outputs, states = rnn_layer(inputs, training=True)
        self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), 3)
        self.assertEqual(len(states[0]), 2)
        self.assertListEqual(states[0][0].shape.as_list(), [4, 20])

    def testLSTM(self):
        lstm = rnn.LSTM(3, 12)
        inputs = tf.random.uniform([4, 5, 5])
        outputs, states = lstm(inputs, training=True)
        self.assertListEqual(outputs.shape.as_list(), [4, 5, 12])
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), 3)

    def testLSTMWithMask(self):
        lstm = rnn.LSTM(3, 12)
        inputs = tf.random.uniform([3, 4, 5])
        lengths = [4, 2, 3]
        mask = tf.sequence_mask(lengths)
        outputs, states = lstm(inputs, mask=mask, training=True)
        last_hidden = states[-1][0]
        for i, length in enumerate(lengths):
            self.assertAllClose(last_hidden[i], outputs[i, length - 1])

    def testBLSTM(self):
        lstm = rnn.LSTM(3, 12, dropout=0.5, bidirectional=True)
        inputs = tf.random.uniform([4, 5, 5])
        outputs, states = lstm(inputs, training=True)
        self.assertListEqual(outputs.shape.as_list(), [4, 5, 24])
        self.assertIsInstance(states, tuple)
        self.assertEqual(len(states), 3)


if __name__ == "__main__":
    tf.test.main()
