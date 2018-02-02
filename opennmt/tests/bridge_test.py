import tensorflow as tf

from opennmt.layers import bridge


def _build_state(num_layers, num_units, batch_size):
  cell = tf.contrib.rnn.MultiRNNCell(
      [tf.contrib.rnn.LSTMCell(num_units) for _ in range(num_layers)])
  return cell.zero_state(batch_size, tf.float32)


class BridgeTest(tf.test.TestCase):

  def testZeroBridge(self):
    encoder_state = _build_state(4, 20, 6)
    decoder_state = _build_state(3, 60, 6)
    state = bridge.ZeroBridge()(encoder_state, decoder_state)
    self.assertTupleEqual(decoder_state, state)

  def testCopyBridge(self):
    encoder_state = _build_state(3, 20, 6)
    decoder_state = _build_state(3, 20, 6)
    state = bridge.CopyBridge()(encoder_state, decoder_state)
    self.assertTupleEqual(encoder_state, state)

  def testCopyBridgeLayerMismatch(self):
    encoder_state = _build_state(3, 20, 6)
    decoder_state = _build_state(4, 20, 6)
    with self.assertRaises(ValueError):
      _ = bridge.CopyBridge()(encoder_state, decoder_state)

  def testCopyBridgeSizeMismatch(self):
    encoder_state = _build_state(3, 20, 6)
    decoder_state = _build_state(3, 30, 6)
    with self.assertRaises(ValueError):
      _ = bridge.CopyBridge()(encoder_state, decoder_state)

  def testDenseBridge(self):
    encoder_state = _build_state(3, 20, 6)
    decoder_state = _build_state(4, 30, 6)
    state = bridge.DenseBridge()(encoder_state, decoder_state)
    bridge.assert_state_is_compatible(decoder_state, state)


if __name__ == "__main__":
  tf.test.main()
