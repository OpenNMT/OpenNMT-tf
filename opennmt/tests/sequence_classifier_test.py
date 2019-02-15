import tensorflow as tf
import numpy as np

from opennmt import encoders
from opennmt.models import sequence_classifier
from opennmt.tests import test_util


@test_util.run_tf1_only
class SequenceClassifierTest(tf.test.TestCase):

  def _testLastEncoding(self, encoder):
    sequence_length = [3, 5, 4]
    batch_size = len(sequence_length)
    input_depth = 5
    x = tf.placeholder_with_default(
        np.random.randn(
            batch_size, max(sequence_length), input_depth).astype(np.float32),
        shape=(None, None, input_depth))
    _, state, _ = encoder.encode(x, sequence_length=sequence_length)
    encoding = sequence_classifier.last_encoding_from_state(state)
    self.assertEqual(2, len(encoding.get_shape().as_list()))
    abs_sum = tf.reduce_sum(tf.abs(encoding), axis=1)
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      abs_sum = sess.run(abs_sum)
      self.assertNotEqual(0, abs_sum[0])
      self.assertNotEqual(0, abs_sum[2])

  def testLastEncoding(self):
    self._testLastEncoding(encoders.UnidirectionalRNNEncoder(
        1, 10, cell_class=tf.nn.rnn_cell.LSTMCell))
    self._testLastEncoding(encoders.UnidirectionalRNNEncoder(
        3, 10, cell_class=tf.nn.rnn_cell.LSTMCell))
    self._testLastEncoding(encoders.UnidirectionalRNNEncoder(
        1, 10, cell_class=tf.nn.rnn_cell.GRUCell))
    self._testLastEncoding(encoders.UnidirectionalRNNEncoder(
        3, 10, cell_class=tf.nn.rnn_cell.GRUCell))


if __name__ == "__main__":
  tf.test.main()
