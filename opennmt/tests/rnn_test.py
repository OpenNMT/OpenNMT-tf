import tensorflow as tf

from opennmt.layers import rnn
from opennmt.layers import reducer
from opennmt.tests import test_util


@test_util.run_tf2_only
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
    rnn_layer = rnn.RNN(cell, bidirectional=True, reducer=reducer.ConcatReducer())
    inputs = tf.random.uniform([4, 5, 5])
    outputs, states = rnn_layer(inputs, training=True)
    self.assertListEqual(outputs.shape.as_list(), [4, 5, 20])
    self.assertIsInstance(states, tuple)
    self.assertEqual(len(states), 3)
    self.assertEqual(len(states[0]), 2)
    self.assertListEqual(states[0][0].shape.as_list(), [4, 20])


if __name__ == "__main__":
  tf.test.main()
