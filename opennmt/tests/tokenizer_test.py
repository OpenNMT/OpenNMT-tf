# -*- coding: utf-8 -*-

import tensorflow as tf

from opennmt.tokenizers import SpaceTokenizer, CharacterTokenizer, OpenNMTTokenizer


class TokenizerTest(tf.test.TestCase):

  def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_bytes(token) for token in ref_tokens]
    text = tf.constant(text)
    tokens = tokenizer.tokenize(text)
    with self.test_session() as sess:
      tokens = sess.run(tokens)
      self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizerOnString(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_text(token) for token in ref_tokens]
    tokens = tokenizer.tokenize(text)
    self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizer(self, tokenizer, text, ref_tokens):
    self._testTokenizerOnTensor(tokenizer, text, ref_tokens)
    self._testTokenizerOnString(tokenizer, text, ref_tokens)

  def _testDetokenizerOnTensor(self, tokenizer, tokens, ref_text):
    ref_text = tf.compat.as_bytes(ref_text)
    tokens = tf.constant(tokens)
    text = tokenizer.detokenize(tokens)
    with self.test_session() as sess:
      text = sess.run(text)
      self.assertEqual(ref_text, text)

  def _testDetokenizerOnBatchTensor(self, tokenizer, tokens, ref_text):
    ref_text = [tf.compat.as_bytes(t) for t in ref_text]
    sequence_length = [len(x) for x in tokens]
    max_length = max(sequence_length)
    tokens = [tok + [""] * (max_length - len(tok)) for tok in tokens]
    tokens = tf.constant(tokens)
    sequence_length = tf.constant(sequence_length)
    text = tokenizer.detokenize(tokens, sequence_length=sequence_length)
    with self.test_session() as sess:
      text = sess.run(text)
      self.assertAllEqual(ref_text, text)

  def _testDetokenizerOnString(self, tokenizer, tokens, ref_text):
    tokens = [tf.compat.as_text(token) for token in tokens]
    ref_text = tf.compat.as_text(ref_text)
    text = tokenizer.detokenize(tokens)
    self.assertAllEqual(ref_text, text)

  def _testDetokenizer(self, tokenizer, tokens, ref_text):
    self._testDetokenizerOnBatchTensor(tokenizer, tokens, ref_text)
    for tok, ref in zip(tokens, ref_text):
      self._testDetokenizerOnTensor(tokenizer, tok, ref)
      self._testDetokenizerOnString(tokenizer, tok, ref)

  def testSpaceTokenizer(self):
    self._testTokenizer(SpaceTokenizer(), "Hello world !", ["Hello", "world", "!"])
    self._testDetokenizer(
        SpaceTokenizer(),
        [["Hello", "world", "!"], ["Test"], ["My", "name"]],
        ["Hello world !", "Test", "My name"])

  def testCharacterTokenizer(self):
    self._testTokenizer(CharacterTokenizer(), "a b", ["a", "▁", "b"])
    self._testDetokenizer(CharacterTokenizer(), [["a", "▁", "b"]], ["a b"])
    self._testTokenizer(CharacterTokenizer(), "你好，世界！", ["你", "好", "，", "世", "界", "！"])

  def testOpenNMTTokenizer(self):
    self._testTokenizer(OpenNMTTokenizer(), "Hello world!", ["Hello", "world", "!"])
    self._testDetokenizer(
        OpenNMTTokenizer(),
        [["Hello", "world", "￭!"], ["Test"], ["My", "name"]],
        ["Hello world!", "Test", "My name"])


if __name__ == "__main__":
  tf.test.main()
