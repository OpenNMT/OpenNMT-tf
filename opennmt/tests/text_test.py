# -*- coding: utf-8 -*-

from parameterized import parameterized

import tensorflow as tf

from opennmt.constants import PADDING_TOKEN as PAD
from opennmt.data import text


class TextTest(tf.test.TestCase):

  def _testTokensToChars(self, tokens, expected_chars, expected_lengths):
    expected_chars = tf.nest.map_structure(tf.compat.as_bytes, expected_chars)
    chars, lengths = text.tokens_to_chars(tf.constant(tokens, dtype=tf.string))
    chars, lengths = self.evaluate([chars, lengths])
    self.assertListEqual(expected_chars, chars.tolist())
    self.assertListEqual(expected_lengths, lengths.tolist())

  def testTokensToCharsEmpty(self):
    self._testTokensToChars([], [], [])

  def testTokensToCharsSingle(self):
    self._testTokensToChars(["Hello"], [["H", "e", "l", "l", "o"]], [5])

  def testTokensToCharsMixed(self):
    self._testTokensToChars(
        ["Just", "a", "测试"],
        [["J", "u", "s", "t"], ["a", PAD, PAD, PAD], ["测", "试", PAD, PAD]],
        [4, 1, 2])

  @parameterized.expand([
    [["a￭", "b", "c￭", "d", "￭e"], [["a￭", "b", ""], ["c￭", "d", "￭e"]]],
    [["a", "￭", "b", "c￭", "d", "￭", "e"], [["a", "￭", "b", ""], ["c￭", "d", "￭", "e"]]],
  ])
  def testToWordsWithJoiner(self, tokens, expected):
    tokens = tf.constant(tokens)
    expected = tf.constant(expected)
    words = text.tokens_to_words(tokens)
    words, expected = self.evaluate([words, expected])
    self.assertAllEqual(words, expected)

  @parameterized.expand([
    [["▁a", "b", "▁c", "d", "e"], [["▁a", "b", ""], ["▁c", "d", "e"]]],
    [["▁", "a", "b", "▁", "c", "d", "e"], [["▁", "a", "b", ""], ["▁", "c", "d", "e"]]],
  ])
  def testToWordsWithSpacer(self, tokens, expected):
    tokens = tf.constant(tokens)
    expected = tf.constant(expected)
    words = text.tokens_to_words(tokens, subword_token="▁", is_spacer=True)
    words, expected = self.evaluate([words, expected])
    self.assertAllEqual(words, expected)


if __name__ == "__main__":
  tf.test.main()
