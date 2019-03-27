import tensorflow as tf
import numpy as np

from opennmt.utils import decoding

class DecodingTest(tf.test.TestCase):

  def testPenalizeToken(self):
    log_probs = tf.zeros([4, 6])
    token_id = 1
    log_probs = decoding._penalize_token(log_probs, token_id)
    log_probs = self.evaluate(log_probs)
    self.assertTrue(np.all(log_probs[:, token_id] < 0))
    non_penalized = np.delete(log_probs, 1, token_id)
    self.assertEqual(np.sum(non_penalized), 0)


if __name__ == "__main__":
  tf.test.main()
