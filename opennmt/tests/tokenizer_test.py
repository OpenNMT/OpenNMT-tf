# -*- coding: utf-8 -*-

import tensorflow as tf

from opennmt.tokenizers import SpaceTokenizer, CharacterTokenizer


class TokenizerTest(tf.test.TestCase):

  def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens):
    text = tf.constant(text)
    tokens = tokenizer(text)
    with self.test_session() as sess:
      tokens = sess.run(tokens)
      tokens = [token.decode("utf-8") for token in tokens]
      self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizerOnString(self, tokenizer, text, ref_tokens):
    tokens = tokenizer(text)
    self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizer(self, tokenizer, text, ref_tokens):
    self._testTokenizerOnTensor(tokenizer, text, ref_tokens)
    self._testTokenizerOnString(tokenizer, text, ref_tokens)

  def testSpaceTokenizer(self):
    self._testTokenizer(
        SpaceTokenizer(),
        u"Hello world !",
        [u"Hello", u"world", u"!"])

  def testCharacterTokenizer(self):
    self._testTokenizer(
        CharacterTokenizer(),
        u"你好，世界！",
        [u"你", u"好", u"，", u"世", u"界", u"！"])


if __name__ == "__main__":
  tf.test.main()
