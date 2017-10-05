# -*- coding: utf-8 -*-

import tensorflow as tf

from opennmt.constants import PADDING_TOKEN as PAD
from opennmt.inputters import text_inputter


class InputterTest(tf.test.TestCase):

  def _testTokensToChars(self, tokens, expected_chars):
    expected_chars = [[tf.compat.as_bytes(c) for c in w] for w in expected_chars]
    chars = text_inputter.tokens_to_chars(tf.constant(tokens))
    with self.test_session() as sess:
      chars = sess.run(chars)
      self.assertAllEqual(expected_chars, chars)

  def testTokensToCharsSingle(self):
    self._testTokensToChars(["Hello"], [["H", "e", "l", "l", "o"]])

  def testTokensToCharsMixed(self):
    self._testTokensToChars(
        ["Just", "a", "测试"],
        [["J", "u", "s", "t"], ["a", PAD, PAD, PAD], ["测", "试", PAD, PAD]])


if __name__ == "__main__":
  tf.test.main()
