# -*- coding: utf-8 -*-

from parameterized import parameterized

import tensorflow as tf

from opennmt.layers import noise
from opennmt.tests import test_util


@test_util.skip_if_unsupported("RaggedTensor")
class NoiseTest(tf.test.TestCase):

  @parameterized.expand([
    [["a￭", "b", "c￭", "d", "￭e"], [["a￭", "b", ""], ["c￭", "d", "￭e"]]],
    [["a", "￭", "b", "c￭", "d", "￭", "e"], [["a", "￭", "b", ""], ["c￭", "d", "￭", "e"]]],
  ])
  def testToWordsWithJoiner(self, tokens, expected):
    tokens = tf.constant(tokens)
    expected = tf.constant(expected)
    words = noise.tokens_to_words(tokens)
    words, expected = self.evaluate([words, expected])
    self.assertAllEqual(words, expected)

  @parameterized.expand([
    [["▁a", "b", "▁c", "d", "e"], [["▁a", "b", ""], ["▁c", "d", "e"]]],
    [["▁", "a", "b", "▁", "c", "d", "e"], [["▁", "a", "b", ""], ["▁", "c", "d", "e"]]],
  ])
  def testToWordsWithSpacer(self, tokens, expected):
    tokens = tf.constant(tokens)
    expected = tf.constant(expected)
    words = noise.tokens_to_words(tokens, subword_token="▁", is_spacer=True)
    words, expected = self.evaluate([words, expected])
    self.assertAllEqual(words, expected)

  @parameterized.expand([
    [["a", "b", "c", "e", "f"]],
    [[["a", "b", ""], ["c", "e", "f"]]],
  ])
  def testWordDropoutNone(self, words):
    x = tf.constant(words)
    y = noise.WordDropout(0)(x)
    x, y = self.evaluate([x, y])
    self.assertAllEqual(x, y)

  @parameterized.expand([
    [["a", "b", "c", "e", "f"]],
    [[["a", "b", ""], ["c", "e", "f"]]],
  ])
  def testWordDropoutAll(self, words):
    x = tf.constant(words)
    y = noise.WordDropout(1)(x)
    y = self.evaluate(y)
    self.assertEqual(y.shape[0], 1)  # At least one is not dropped.

  @parameterized.expand([
    [["a", "b", "c"], ["d", "d", "d"]],
    [[["a", "b", ""], ["c", "e", "f"]], [["d", "", ""], ["d", "", ""]]],
  ])
  def testWordReplacement(self, words, expected):
    expected = tf.constant(expected)
    words = tf.constant(words)
    words = noise.WordReplacement(1, filler="d")(words)
    words, expected = self.evaluate([words, expected])
    self.assertAllEqual(words, expected)

  @parameterized.expand([[0], [1], [3]])
  def testWordPermutation(self, k):
    x = tf.constant("0 1 2 3 4 5 6 7 8 9 10 11 12 13".split())
    y = noise.WordPermutation(k)(x)
    x, y = self.evaluate([x, y])
    if k == 0:
      self.assertAllEqual(y, x)
    else:
      for i, v in enumerate(y.tolist()):
        self.assertLess(abs(int(v) - i), k)

  def testWordNoising(self):
    tokens = tf.constant([["a￭", "b", "c￭", "d", "￭e"], ["a", "b", "c", "", ""]])
    lengths = tf.constant([5, 3])
    noiser = noise.WordNoiser()
    noiser.add(noise.WordDropout(0.1))
    noiser.add(noise.WordReplacement(0.1))
    noiser.add(noise.WordPermutation(3))
    noisy_tokens, noisy_lengths = noiser(tokens, sequence_length=lengths)
    tokens, noisy_tokens = self.evaluate([tokens, noisy_tokens])
    self.assertAllEqual(noisy_tokens.shape, tokens.shape)


if __name__ == "__main__":
  tf.test.main()
