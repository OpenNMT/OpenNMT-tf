import os

import tensorflow as tf

from opennmt.inputters import text_inputter
from opennmt.models import sequence_to_sequence
from opennmt.tests import test_util


class SequenceToSequenceTest(tf.test.TestCase):
    def testReplaceUnknownTarget(self):
        target_tokens = [
            ["Hello", "world", "!", "", "", ""],
            ["<unk>", "name", "is", "<unk>", ".", ""],
        ]
        source_tokens = [
            ["Bonjour", "le", "monde", "!", ""],
            ["Mon", "nom", "est", "Max", "."],
        ]
        attention = [
            [
                [0.9, 0.1, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.7, 0.0, 0.0],
                [0.0, 0.1, 0.1, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.8, 0.1, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.2, 0.6, 0.0],
                [0.0, 0.1, 0.1, 0.3, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
        replaced_target_tokens = sequence_to_sequence.replace_unknown_target(
            target_tokens, source_tokens, attention, unknown_token="<unk>"
        )
        replaced_target_tokens = self.evaluate(replaced_target_tokens)
        self.assertNotIn(b"<unk>", replaced_target_tokens.flatten().tolist())
        self.assertListEqual(
            [b"Hello", b"world", b"!", b"", b"", b""],
            replaced_target_tokens[0].tolist(),
        )
        self.assertListEqual(
            [b"Mon", b"name", b"is", b"Max", b".", b""],
            replaced_target_tokens[1].tolist(),
        )

    def testMaskAttention(self):
        attention = [
            [
                [0.9, 0.1, 0.0, 0.0, 0.0],
                [0.2, 0.1, 0.7, 0.0, 0.0],
                [0.0, 0.1, 0.1, 0.8, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.8, 0.1, 0.1, 0.0, 0.0],
                [0.1, 0.9, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.8, 0.1, 0.0],
                [0.1, 0.1, 0.2, 0.6, 0.0],
                [0.0, 0.1, 0.1, 0.3, 0.5],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]
        self.assertAllClose(
            sequence_to_sequence.mask_attention(attention, [2, 3], True, True),
            [
                [
                    [0.1, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.7, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.1, 0.1, 0.0, 0.0, 0.0],
                    [0.9, 0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.8, 0.1, 0.0, 0.0],
                    [0.1, 0.2, 0.6, 0.0, 0.0],
                    [0.1, 0.1, 0.3, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ],
        )

    def testSequenceToSequenceInputter(self):
        source_vocabulary = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src_vocab.txt"),
            ["<blank>", "<s>", "</s>", "a", "b", "c", "d"],
        )
        target_vocabulary = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "tgt_vocab.txt"),
            ["<blank>", "<s>", "</s>", "e", "f", "g", "h"],
        )
        source_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src.txt"), ["a c c", "b d", "a e"]
        )
        target_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "tgt.txt"), ["f h g", "e h", "a e"]
        )
        inputter = sequence_to_sequence.SequenceToSequenceInputter(
            text_inputter.WordEmbedder(embedding_size=20),
            text_inputter.WordEmbedder(embedding_size=20),
        )
        inputter.initialize(
            dict(
                source_vocabulary=source_vocabulary, target_vocabulary=target_vocabulary
            )
        )
        dataset = inputter.make_dataset([source_file, target_file])
        element = iter(dataset).next()
        features, labels = inputter.make_features(element)
        self.assertIn("ids_out", labels)
        self.assertAllEqual(labels["ids"], [1, 4, 6, 5])
        self.assertAllEqual(labels["ids_out"], [4, 6, 5, 2])
        self.assertEqual(labels["length"], 4)


if __name__ == "__main__":
    tf.test.main()
