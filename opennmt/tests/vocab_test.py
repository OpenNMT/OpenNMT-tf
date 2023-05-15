import os

import numpy as np
import tensorflow as tf

from opennmt.data import vocab as vocab_lib
from opennmt.optimizers.utils import make_optimizer
from opennmt.tests import test_util


class VocabTest(tf.test.TestCase):
    def testSimpleVocab(self):
        vocab = vocab_lib.Vocab()

        self.assertEqual(0, vocab.size)

        vocab.add("toto")
        vocab.add("toto")
        vocab.add("toto")
        vocab.add("titi")
        vocab.add("titi")
        vocab.add("tata")

        self.assertEqual(3, vocab.size)
        self.assertEqual(1, vocab.lookup("titi"))
        self.assertEqual("titi", vocab.lookup(1))

        pruned_size = vocab.prune(max_size=2)

        self.assertEqual(2, pruned_size.size)
        self.assertEqual(None, pruned_size.lookup("tata"))

        pruned_frequency = vocab.prune(min_frequency=3)

        self.assertEqual(1, pruned_frequency.size)
        self.assertEqual(0, pruned_frequency.lookup("toto"))

    def testVocabWithSpecialTokens(self):
        vocab = vocab_lib.Vocab(special_tokens=["foo", "bar"])

        self.assertEqual(2, vocab.size)

        vocab.add("toto")
        vocab.add("toto")
        vocab.add("toto")
        vocab.add("titi")
        vocab.add("titi")
        vocab.add("tata")

        self.assertEqual(5, vocab.size)
        self.assertEqual(3, vocab.lookup("titi"))

        pruned_size = vocab.prune(max_size=3)

        self.assertEqual(3, pruned_size.size)
        self.assertEqual(0, pruned_size.lookup("foo"))
        self.assertEqual(1, pruned_size.lookup("bar"))

    def testVocabSaveAndLoad(self):
        vocab1 = vocab_lib.Vocab(special_tokens=["foo", "bar"])
        vocab1.add("toto")
        vocab1.add("toto")
        vocab1.add("toto")
        vocab1.add("titi")
        vocab1.add("titi")
        vocab1.add("tata")

        vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")

        vocab1.serialize(vocab_file)
        vocab2 = vocab_lib.Vocab.from_file(vocab_file)

        self.assertEqual(vocab1.size, vocab2.size)
        self.assertEqual(vocab1.lookup("titi"), vocab2.lookup("titi"))

    def testLoadSentencePieceVocab(self):
        vocab_path = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "vocab_sp"),
            [
                "<unk>	0",
                "<s>	0",
                "</s>	0",
                ",	-3.0326",
                ".	-3.41093",
                "▁the	-3.85169",
                "s	-4.05468",
                "▁die	-4.15914",
                "▁in	-4.2419",
                "▁der	-4.36135",
            ],
        )

        vocab = vocab_lib.Vocab.from_file(vocab_path, file_format="sentencepiece")
        self.assertEqual(len(vocab), 7)
        self.assertNotIn("<unk>", vocab)
        self.assertNotIn("<s>", vocab)
        self.assertNotIn("</s>", vocab)
        self.assertIn("▁the", vocab)

    def testVocabPadding(self):
        vocab = vocab_lib.Vocab()
        vocab.add("toto")
        vocab.add("titi")
        vocab.add("tata")
        self.assertEqual(vocab.size, 3)
        vocab.pad_to_multiple(6, num_oov_buckets=1)
        self.assertEqual(vocab.size, 6 - 1)

    def testVocabNoPadding(self):
        vocab = vocab_lib.Vocab()
        vocab.add("toto")
        vocab.add("titi")
        vocab.add("tata")
        self.assertEqual(vocab.size, 3)
        vocab.pad_to_multiple(4, num_oov_buckets=1)
        self.assertEqual(vocab.size, 3)

    def _saveVocab(self, name, words):
        vocab = vocab_lib.Vocab()
        for word in words:
            vocab.add(str(word))
        vocab_file = os.path.join(self.get_temp_dir(), name)
        vocab.serialize(vocab_file)
        return vocab_file

    def testVocabMappingMerge(self):
        old = self._saveVocab("old", ["1", "2", "3", "4"])
        new = self._saveVocab("new", ["1", "6", "3", "5", "7"])
        mapping, new_vocab = vocab_lib.get_mapping(old, new, "merge")
        self.assertEqual(4 + 5 - 2 + 1, len(mapping))  # old + new - common + <unk>
        self.assertAllEqual([0, 1, 2, 3, -1, -1, -1, 4], mapping)
        self.assertAllEqual(["1", "2", "3", "4", "6", "5", "7"], new_vocab.words)

    def testVocabMappingReplace(self):
        old = self._saveVocab("old", ["1", "2", "3", "4"])
        new = self._saveVocab("new", ["1", "6", "5", "3", "7"])
        mapping, new_vocab = vocab_lib.get_mapping(old, new, "replace")
        self.assertEqual(5 + 1, len(mapping))  # new + <unk>
        self.assertAllEqual([0, -1, -1, 2, -1, 4], mapping)
        self.assertAllEqual(["1", "6", "5", "3", "7"], new_vocab.words)

    def testVocabVariableUpdate(self):
        ref_variable, ref_optimizer = _create_variable_and_slots([1, 2, 3, 4, 5, 6, 7])
        new_variable, new_optimizer = _create_variable_and_slots([0, 0, 0, 0, 0, 0])
        mapping = [0, -1, -1, 4, -1, 2]
        expected = [1, 0, 0, 5, 0, 3]
        vocab_lib.update_variable_and_slots(
            ref_variable, new_variable, ref_optimizer, new_optimizer, mapping
        )
        variables = [new_variable] + [
            new_optimizer.get_slot(new_variable, slot) for slot in ("m", "v")
        ]
        for variable in self.evaluate(variables):
            self.assertAllEqual(variable, expected)


def _create_variable_and_slots(values):
    variable = tf.Variable(tf.constant(values, dtype=tf.float32))
    optimizer = make_optimizer("Adam", 0.001)
    optimizer._create_slots([variable])
    for slot in ("m", "v"):
        optimizer.get_slot(variable, slot).assign(variable)
    return variable, optimizer


if __name__ == "__main__":
    tf.test.main()
