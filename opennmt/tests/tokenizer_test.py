# -*- coding: utf-8 -*-

import os
import yaml

import tensorflow as tf

from opennmt.tokenizers import SpaceTokenizer, CharacterTokenizer, OpenNMTTokenizer


class TokenizerTest(tf.test.TestCase):

  def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_bytes(token) for token in ref_tokens]
    text = tf.constant(text)
    tokens = tokenizer.tokenize(text)
    self.assertAllEqual(ref_tokens, self.evaluate(tokens))

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
    self.assertEqual(ref_text, self.evaluate(text))

  def _testDetokenizerOnBatchTensor(self, tokenizer, tokens, ref_text):
    ref_text = [tf.compat.as_bytes(t) for t in ref_text]
    sequence_length = [len(x) for x in tokens]
    max_length = max(sequence_length)
    tokens = [tok + [""] * (max_length - len(tok)) for tok in tokens]
    tokens = tf.constant(tokens)
    sequence_length = tf.constant(sequence_length)
    text = tokenizer.detokenize(tokens, sequence_length=sequence_length)
    self.assertAllEqual(ref_text, self.evaluate(text))

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

  def testOpenNMTTokenizerFromConfiguration(self):
    params = {
        "mode": "aggressive",
        "spacer_annotate": True,
        "spacer_new": True
    }
    tok_config = os.path.join(self.get_temp_dir(), "tok_config.yml")
    with open(tok_config, "w") as tok_config_file:
      yaml.dump(params, tok_config_file)

    def _test(tokenizer):
      self._testTokenizer(tokenizer, "Hello World-s", ["Hello", "▁", "World", "-", "s"])

    tokenizer = OpenNMTTokenizer(configuration_file_or_key=tok_config)
    _test(tokenizer)
    tokenizer = OpenNMTTokenizer(configuration_file_or_key="source_tokenization")
    tokenizer.initialize({"source_tokenization": tok_config})
    _test(tokenizer)
    tokenizer = OpenNMTTokenizer(configuration_file_or_key="source_tokenization")
    tokenizer.initialize({"source_tokenization": params})
    _test(tokenizer)
    tokenizer = OpenNMTTokenizer(params=params)
    _test(tokenizer)

  def testOpenNMTTokenizerAssets(self):
    asset_dir = self.get_temp_dir()
    # Write a dummy BPE model.
    bpe_model_path = os.path.join(asset_dir, "model.bpe")
    with open(bpe_model_path, "wb") as bpe_model_file:
      bpe_model_file.write(b"#version: 0.2\ne s</w>\n")

    tokenizer = OpenNMTTokenizer(
        params={"mode": "conservative", "bpe_model_path": bpe_model_path})

    # By default, no assets are returned.
    assets = tokenizer.initialize({})
    self.assertDictEqual(assets, {})

    # Generated assets are prefixed but not existing resources.
    assets = tokenizer.initialize({}, asset_dir=asset_dir, asset_prefix="source_")
    self.assertIn("source_tokenizer_config.yml", assets)
    self.assertTrue(os.path.exists(assets["source_tokenizer_config.yml"]))
    self.assertIn("model.bpe", assets)
    self.assertTrue(os.path.exists(assets["model.bpe"]))

    # The tokenization configuration should not contain absolute paths to resources.
    with open(assets["source_tokenizer_config.yml"], "rb") as config_file:
      asset_config = yaml.load(config_file.read(), Loader=yaml.UnsafeLoader)
    self.assertDictEqual(asset_config, {"mode": "conservative", "bpe_model_path": "model.bpe"})


if __name__ == "__main__":
  tf.test.main()
