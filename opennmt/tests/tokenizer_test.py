# -*- coding: utf-8 -*-

import tensorflow as tf

from opennmt.tokenizers import SpaceTokenizer, CharacterTokenizer


class TokenizerTest(tf.test.TestCase):

  def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_bytes(token) for token in ref_tokens]
    text = tf.constant(text)
    tokens = tokenizer(text)
    with self.test_session() as sess:
      tokens = sess.run(tokens)
      self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizerOnString(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_text(token) for token in ref_tokens]
    tokens = tokenizer(text)
    self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizer(self, tokenizer, text, ref_tokens):
    self._testTokenizerOnTensor(tokenizer, text, ref_tokens)
    self._testTokenizerOnString(tokenizer, text, ref_tokens)

  def testSpaceTokenizer(self):
    self._testTokenizer(SpaceTokenizer(), "Hello world !", ["Hello", "world", "!"])

  def testCharacterTokenizer(self):
    self._testTokenizer(CharacterTokenizer(), "a b", ["a", " ", "b"])
    self._testTokenizer(CharacterTokenizer(), "你好，世界！", ["你", "好", "，", "世", "界", "！"])


if __name__ == "__main__":
  tf.test.main()
