import os
import shutil
import unittest

import tensorflow as tf
import yaml

from opennmt import tokenizers
from opennmt.tests import test_util

sp_model = os.path.join(test_util.get_test_data_dir(), "wmtende.model")


class TokenizerTest(tf.test.TestCase):
    def _testTokenizerOnTensor(self, tokenizer, text, ref_tokens, training):
        ref_tokens = [tf.compat.as_bytes(token) for token in ref_tokens]
        text = tf.constant(text)
        tokens = tokenizer.tokenize(text, training=training)
        self.assertIsInstance(tokens, tf.Tensor)
        if not ref_tokens:
            self.assertEmpty(tokens)
        else:
            self.assertAllEqual(ref_tokens, tokens)

    def _testTokenizerOnBatchTensor(self, tokenizer, text, ref_tokens, training):
        text = tf.constant(text)
        tokens = tokenizer.tokenize(text, training=training)
        self.assertIsInstance(tokens, tf.RaggedTensor)
        self.assertListEqual(
            tokens.to_list(), tf.nest.map_structure(tf.compat.as_bytes, ref_tokens)
        )

    def _testTokenizerOnStream(self, tokenizer, text, ref_tokens, training):
        input_path = os.path.join(self.get_temp_dir(), "input.txt")
        output_path = os.path.join(self.get_temp_dir(), "output.txt")
        with open(input_path, "w") as input_file:
            for line in text:
                input_file.write(line)
                input_file.write("\n")
        with open(input_path) as input_file, open(output_path, "w") as output_file:
            tokenizer.tokenize_stream(input_file, output_file, training=training)
        with open(output_path) as output_file:
            for output, ref in zip(output_file, ref_tokens):
                self.assertEqual(output.strip().split(), ref)

    def _testTokenizerOnString(self, tokenizer, text, ref_tokens, training):
        tokens = tokenizer.tokenize(text, training=training)
        self.assertAllEqual(ref_tokens, tokens)

    def _testTokenizer(self, tokenizer, text, ref_tokens, training=True):
        self.assertListEqual(tokenizer.tokenize(text, training), ref_tokens)
        self._testTokenizerOnBatchTensor(tokenizer, text, ref_tokens, training)
        self._testTokenizerOnStream(tokenizer, text, ref_tokens, training)
        for txt, ref in zip(text, ref_tokens):
            self._testTokenizerOnTensor(tokenizer, txt, ref, training)
            self._testTokenizerOnString(tokenizer, txt, ref, training)

    def _testDetokenizerOnTensor(self, tokenizer, tokens, ref_text):
        ref_text = tf.compat.as_bytes(ref_text)
        tokens = tf.constant(tokens, dtype=tf.string)
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
        self.assertAllEqual(
            self.evaluate(text), tf.nest.map_structure(tf.compat.as_bytes, ref_text)
        )

    def _testDetokenizerOnStream(self, tokenizer, tokens, ref_text):
        input_path = os.path.join(self.get_temp_dir(), "input.txt")
        output_path = os.path.join(self.get_temp_dir(), "output.txt")
        with open(input_path, "w") as input_file:
            for tok in tokens:
                input_file.write(" ".join(tok))
                input_file.write("\n")
        with open(input_path) as input_file, open(output_path, "w") as output_file:
            tokenizer.detokenize_stream(input_file, output_file)
        with open(output_path) as output_file:
            for output, ref in zip(output_file, ref_text):
                self.assertEqual(output.strip(), ref)

    def _testDetokenizerOnString(self, tokenizer, tokens, ref_text):
        text = tokenizer.detokenize(tokens)
        self.assertAllEqual(ref_text, text)

    def _testDetokenizer(self, tokenizer, tokens, ref_text):
        self.assertAllEqual(tokenizer.detokenize(tokens), ref_text)
        self._testDetokenizerOnBatchTensor(tokenizer, tokens, ref_text)
        self._testDetokenizerOnBatchRaggedTensor(tokenizer, tokens, ref_text)
        self._testDetokenizerOnStream(tokenizer, tokens, ref_text)
        for tok, ref in zip(tokens, ref_text):
            self._testDetokenizerOnTensor(tokenizer, tok, ref)
            self._testDetokenizerOnString(tokenizer, tok, ref)

    def testSpaceTokenizer(self):
        self._testTokenizer(
            tokenizers.SpaceTokenizer(),
            ["Hello world !", "How are you ?", "", "Good !"],
            [["Hello", "world", "!"], ["How", "are", "you", "?"], [], ["Good", "!"]],
        )
        self._testDetokenizer(
            tokenizers.SpaceTokenizer(),
            [["Hello", "world", "!"], ["Test"], [], ["My", "name"]],
            ["Hello world !", "Test", "", "My name"],
        )

    def testCharacterTokenizer(self):
        self._testTokenizer(
            tokenizers.CharacterTokenizer(),
            ["a b", "", "cd e"],
            [["a", "▁", "b"], [], ["c", "d", "▁", "e"]],
        )
        self._testDetokenizer(
            tokenizers.CharacterTokenizer(),
            [["a", "▁", "b"], [], ["c", "d", "▁", "e"]],
            ["a b", "", "cd e"],
        )
        self._testTokenizer(
            tokenizers.CharacterTokenizer(),
            ["你好，世界！"],
            [["你", "好", "，", "世", "界", "！"]],
        )

    @unittest.skipIf(not os.path.isfile(sp_model), "Missing SentencePiece test model")
    def testSentencePieceTokenizer(self):
        tokenizer_class = getattr(tokenizers, "SentencePieceTokenizer", None)
        if tokenizer_class is None:
            self.skipTest("tensorflow-text is not installed")
        tokenizer = tokenizer_class(sp_model)
        text = ["Hello world!", "", "How are you?"]
        tokens = [["▁H", "ello", "▁world", "!"], [], ["▁How", "▁are", "▁you", "?"]]
        self._testTokenizer(tokenizer, text, tokens)
        self._testDetokenizer(tokenizer, tokens, text)

        # Test SavedModel export.
        class _InputLayer(tf.Module):
            def __init__(self, model_path):
                self._tokenizer = tokenizer_class(model_path)

            @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string)])
            def call(self, text):
                return self._tokenizer.tokenize(text).to_tensor()

        # Use a temporary model file to make sure the exported model is self-contained.
        tmp_model_path = os.path.join(self.get_temp_dir(), "sp.model")
        shutil.copyfile(sp_model, tmp_model_path)
        layer = _InputLayer(tmp_model_path)
        export_dir = os.path.join(self.get_temp_dir(), "export")
        tf.saved_model.save(layer, export_dir)
        os.remove(tmp_model_path)

        imported = tf.saved_model.load(export_dir)
        func = imported.signatures["serving_default"]
        outputs = func(tf.constant(text))["output_0"]
        outputs = tf.RaggedTensor.from_tensor(outputs, padding="").to_list()
        self.assertListEqual(outputs, tf.nest.map_structure(tf.compat.as_bytes, tokens))

    def testOpenNMTTokenizer(self):
        self._testTokenizer(
            tokenizers.OpenNMTTokenizer(),
            ["Hello world!", "", "How are you?"],
            [["Hello", "world", "!"], [], ["How", "are", "you", "?"]],
        )
        self._testDetokenizer(
            tokenizers.OpenNMTTokenizer(),
            [["Hello", "world", "￭!"], ["Test"], [], ["My", "name"]],
            ["Hello world!", "Test", "", "My name"],
        )

    def testOpenNMTTokenizerInFunction(self):
        tokenizer = tokenizers.OpenNMTTokenizer()

        @tf.function
        def _tokenize(text):
            return tokenizer.tokenize(text)

        tokens = _tokenize(tf.constant("Hello world!"))
        self.assertAllEqual(self.evaluate(tokens), [b"Hello", b"world", b"!"])

    def testOpenNMTTokenizerArguments(self):
        with self.assertRaises(ValueError):
            tokenizers.OpenNMTTokenizer(case_feature=True)
        tokenizer = tokenizers.OpenNMTTokenizer(
            mode="aggressive", spacer_annotate=True, spacer_new=True, case_feature=False
        )
        self._testTokenizer(
            tokenizer, ["Hello World-s"], [["Hello", "▁", "World", "-", "s"]]
        )

    @unittest.skipIf(not os.path.isfile(sp_model), "Missing SentencePiece model")
    def testOpenNMTTokenizerInferenceMode(self):
        tokenizer = tokenizers.OpenNMTTokenizer(
            mode="none",
            sp_model_path=sp_model,
            sp_nbest_size=64,
            sp_alpha=0.1,
        )
        self._testTokenizer(tokenizer, ["appealing"], [["▁appealing"]], training=False)

    def testOpenNMTTokenizerAssets(self):
        asset_dir = self.get_temp_dir()
        # Write a dummy BPE model.
        bpe_model_path = os.path.join(asset_dir, "model.bpe")
        with open(bpe_model_path, "w") as bpe_model_file:
            bpe_model_file.write("#version: 0.2\ne s</w>\n")

        tokenizer = tokenizers.OpenNMTTokenizer(
            mode="conservative", bpe_model_path=bpe_model_path
        )

        # Generated assets are prefixed but not existing resources.
        assets = tokenizer.export_assets(asset_dir, asset_prefix="source_")
        self.assertIn("source_tokenizer_config.yml", assets)
        self.assertTrue(os.path.exists(assets["source_tokenizer_config.yml"]))
        self.assertIn("model.bpe", assets)
        self.assertTrue(os.path.exists(assets["model.bpe"]))

        # The tokenization configuration should not contain absolute paths to resources.
        with open(assets["source_tokenizer_config.yml"]) as config_file:
            asset_config = yaml.safe_load(config_file.read())
        self.assertDictEqual(
            asset_config, {"mode": "conservative", "bpe_model_path": "model.bpe"}
        )

    def testMakeTokenizer(self):
        tokenizer = tokenizers.make_tokenizer()
        self.assertIsInstance(tokenizer, tokenizers.SpaceTokenizer)
        self.assertFalse(tokenizer.in_graph)
        tokenizer = tokenizers.make_tokenizer({"type": "SpaceTokenizer"})
        self.assertIsInstance(tokenizer, tokenizers.SpaceTokenizer)
        self.assertTrue(tokenizer.in_graph)
        self.assertIsInstance(
            tokenizers.make_tokenizer({"mode": "conservative"}),
            tokenizers.OpenNMTTokenizer,
        )
        self.assertIsInstance(
            tokenizers.make_tokenizer('{"mode": "conservative"}'),
            tokenizers.OpenNMTTokenizer,
        )
        self.assertIsInstance(
            tokenizers.make_tokenizer(
                {"type": "OpenNMTTokenizer", "params": {"mode": "conservative"}}
            ),
            tokenizers.OpenNMTTokenizer,
        )
        config_path = os.path.join(self.get_temp_dir(), "tok_config.yml")
        with open(config_path, "w") as config_file:
            yaml.dump({"mode": "conservative"}, config_file)
        self.assertIsInstance(
            tokenizers.make_tokenizer(config_path), tokenizers.OpenNMTTokenizer
        )
        with self.assertRaisesRegex(ValueError, "is not in list of"):
            tokenizers.make_tokenizer({"type": "UnknownTokenizer"})
        with self.assertRaisesRegex(ValueError, "is not in list of"):
            tokenizers.make_tokenizer({"type": "Tokenizer"})


if __name__ == "__main__":
    tf.test.main()
