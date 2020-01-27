# -*- coding: utf-8 -*-

import os
import yaml

import tensorflow as tf

from opennmt import tokenizers


class TokenizerTest(tf.test.TestCase):

  def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens):
    ref_tokens = [tf.compat.as_bytes(token) for token in ref_tokens]
    text = tf.constant(text)
    tokens = tokenizer.tokenize(text)
    self.assertIsInstance(tokens, tf.Tensor)
    self.assertAllEqual(ref_tokens, self.evaluate(tokens))

  def _testTokenizerOnBatchTensor(self, tokenizer, text, ref_tokens):
    text = tf.constant(text)
    tokens = tokenizer.tokenize(text)
    self.assertIsInstance(tokens, tf.RaggedTensor)
    self.assertAllEqual(tokens.to_list(), tf.nest.map_structure(tf.compat.as_bytes, ref_tokens))

  def _testTokenizerOnString(self, tokenizer, text, ref_tokens):
    tokens = tokenizer.tokenize(text)
    self.assertAllEqual(ref_tokens, tokens)

  def _testTokenizer(self, tokenizer, text, ref_tokens):
    self.assertAllEqual(tokenizer.tokenize(text), ref_tokens)
    self._testTokenizerOnBatchTensor(tokenizer, text, ref_tokens)
    for txt, ref in zip(text, ref_tokens):
      self._testTokenizerOnTensor(tokenizer, txt, ref)
      self._testTokenizerOnString(tokenizer, txt, ref)

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

  def _testDetokenizerOnBatchRaggedTensor(self, tokenizer, tokens, ref_text):
    lengths = tf.constant([len(x) for x in tokens])
    flat_tokens = tf.constant(tf.nest.flatten(tokens))
    ragged_tokens = tf.RaggedTensor.from_row_lengths(flat_tokens, lengths)
    text = tokenizer.detokenize(ragged_tokens)
    self.assertAllEqual(self.evaluate(text), tf.nest.map_structure(tf.compat.as_bytes, ref_text))

  def _testDetokenizerOnString(self, tokenizer, tokens, ref_text):
    text = tokenizer.detokenize(tokens)
    self.assertAllEqual(ref_text, text)

  def _testDetokenizer(self, tokenizer, tokens, ref_text):
    self.assertAllEqual(tokenizer.detokenize(tokens), ref_text)
    self._testDetokenizerOnBatchTensor(tokenizer, tokens, ref_text)
    self._testDetokenizerOnBatchRaggedTensor(tokenizer, tokens, ref_text)
    for tok, ref in zip(tokens, ref_text):
      self._testDetokenizerOnTensor(tokenizer, tok, ref)
      self._testDetokenizerOnString(tokenizer, tok, ref)

  def testSpaceTokenizer(self):
    self._testTokenizer(
        tokenizers.SpaceTokenizer(),
        ["Hello world !", "How are you ?", "Good !"],
        [["Hello", "world", "!"], ["How", "are", "you", "?"], ["Good", "!"]])
    self._testDetokenizer(
        tokenizers.SpaceTokenizer(),
        [["Hello", "world", "!"], ["Test"], ["My", "name"]],
        ["Hello world !", "Test", "My name"])

  def testCharacterTokenizer(self):
    self._testTokenizer(
        tokenizers.CharacterTokenizer(),
        ["a b", "cd e"],
        [["a", "▁", "b"], ["c", "d", "▁", "e"]])
    self._testDetokenizer(
        tokenizers.CharacterTokenizer(),
        [["a", "▁", "b"], ["c", "d", "▁", "e"]],
        ["a b", "cd e"])
    self._testTokenizer(
        tokenizers.CharacterTokenizer(),
        ["你好，世界！"],
        [["你", "好", "，", "世", "界", "！"]])

  def testOpenNMTTokenizer(self):
    self._testTokenizer(
        tokenizers.OpenNMTTokenizer(),
        ["Hello world!", "How are you?"],
        [["Hello", "world", "!"], ["How", "are", "you", "?"]])
    self._testDetokenizer(
        tokenizers.OpenNMTTokenizer(),
        [["Hello", "world", "￭!"], ["Test"], ["My", "name"]],
        ["Hello world!", "Test", "My name"])

  def testOpenNMTTokenizerArguments(self):
    tokenizer = tokenizers.OpenNMTTokenizer(
        mode="aggressive", spacer_annotate=True, spacer_new=True)
    self._testTokenizer(tokenizer, ["Hello World-s"], [["Hello", "▁", "World", "-", "s"]])

  def testOpenNMTTokenizerAssets(self):
    asset_dir = self.get_temp_dir()
    # Write a dummy BPE model.
    bpe_model_path = os.path.join(asset_dir, "model.bpe")
    with open(bpe_model_path, "w") as bpe_model_file:
      bpe_model_file.write("#version: 0.2\ne s</w>\n")

    tokenizer = tokenizers.OpenNMTTokenizer(mode="conservative", bpe_model_path=bpe_model_path)

    # Generated assets are prefixed but not existing resources.
    assets = tokenizer.export_assets(asset_dir, asset_prefix="source_")
    self.assertIn("source_tokenizer_config.yml", assets)
    self.assertTrue(os.path.exists(assets["source_tokenizer_config.yml"]))
    self.assertIn("model.bpe", assets)
    self.assertTrue(os.path.exists(assets["model.bpe"]))

    # The tokenization configuration should not contain absolute paths to resources.
    with open(assets["source_tokenizer_config.yml"], "rb") as config_file:
      asset_config = yaml.load(config_file.read(), Loader=yaml.UnsafeLoader)
    self.assertDictEqual(asset_config, {"mode": "conservative", "bpe_model_path": "model.bpe"})

  def testMakeTokenizer(self):
    tokenizer = tokenizers.make_tokenizer()
    self.assertIsInstance(tokenizer, tokenizers.SpaceTokenizer)
    self.assertFalse(tokenizer.in_graph)
    tokenizer = tokenizers.make_tokenizer({"type": "SpaceTokenizer"})
    self.assertIsInstance(tokenizer, tokenizers.SpaceTokenizer)
    self.assertTrue(tokenizer.in_graph)
    self.assertIsInstance(
        tokenizers.make_tokenizer({"mode": "conservative"}),
        tokenizers.OpenNMTTokenizer)
    self.assertIsInstance(
        tokenizers.make_tokenizer({"type": "OpenNMTTokenizer", "params": {"mode": "conservative"}}),
        tokenizers.OpenNMTTokenizer)
    config_path = os.path.join(self.get_temp_dir(), "tok_config.yml")
    with open(config_path, "w") as config_file:
      yaml.dump({"mode": "conservative"}, config_file)
    self.assertIsInstance(
        tokenizers.make_tokenizer(config_path),
        tokenizers.OpenNMTTokenizer)
    with self.assertRaisesRegex(ValueError, "is not in list of"):
      tokenizers.make_tokenizer({"type": "UnknownTokenizer"})
    with self.assertRaisesRegex(ValueError, "is not in list of"):
      tokenizers.make_tokenizer({"type": "Tokenizer"})


if __name__ == "__main__":
  tf.test.main()
