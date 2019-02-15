import os

import tensorflow as tf

from opennmt import constants
from opennmt.models import sequence_to_sequence
from opennmt.inputters.text_inputter import WordEmbedder
from opennmt.tests import test_util


class SequenceToSequenceTest(tf.test.TestCase):

  @test_util.run_tf1_only
  def testShiftTargetSequenceHook(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    with open(vocab_file, "wb") as vocab:
      vocab.write(b"<blank>\n"
                  b"<s>\n"
                  b"</s>\n"
                  b"the\n"
                  b"world\n"
                  b"hello\n"
                  b"toto\n")
    inputter = WordEmbedder("vocabulary_file", embedding_size=10)
    inputter.add_process_hooks([sequence_to_sequence.shift_target_sequence])
    inputter.initialize({"vocabulary_file": vocab_file})
    data = inputter.process(tf.constant("hello world !"))
    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      data = sess.run(data)
      self.assertAllEqual(data["ids"], [1, 5, 4, 7])
      self.assertAllEqual(data["ids_out"], [5, 4, 7, 2])
      self.assertEqual(data["length"], 4)

  def testReplaceUnknownTarget(self):
    target_tokens = [
      ["Hello", "world", "!", "", "", ""],
      ["<unk>", "name", "is", "<unk>", ".", ""]]
    source_tokens = [
      ["Bonjour", "le", "monde", "!", ""],
      ["Mon", "nom", "est", "Max", "."]]
    attention = [
      [[0.9, 0.1, 0.0, 0.0, 0.0],
       [0.2, 0.1, 0.7, 0.0, 0.0],
       [0.0, 0.1, 0.1, 0.8, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0],
       [0.0, 0.0, 0.0, 0.0, 0.0]],
      [[0.8, 0.1, 0.1, 0.0, 0.0],
       [0.1, 0.9, 0.0, 0.0, 0.0],
       [0.0, 0.1, 0.8, 0.1, 0.0],
       [0.1, 0.1, 0.2, 0.6, 0.0],
       [0.0, 0.1, 0.1, 0.3, 0.5],
       [0.0, 0.0, 0.0, 0.0, 0.0]]]
    replaced_target_tokens = sequence_to_sequence.replace_unknown_target(
        target_tokens,
        source_tokens,
        attention,
        unknown_token="<unk>")
    replaced_target_tokens = self.evaluate(replaced_target_tokens)
    self.assertNotIn(b"<unk>", replaced_target_tokens.flatten().tolist())
    self.assertListEqual(
        [b"Hello", b"world", b"!", b"", b"", b""], replaced_target_tokens[0].tolist())
    self.assertListEqual(
        [b"Mon", b"name", b"is", b"Max", b".", b""], replaced_target_tokens[1].tolist())

  def _testPharaohAlignments(self, line, lengths, expected_matrix):
    matrix = sequence_to_sequence.alignment_matrix_from_pharaoh(
        tf.constant(line), lengths[0], lengths[1], dtype=tf.int32)
    self.assertListEqual(expected_matrix, self.evaluate(matrix).tolist())

  def testPharaohAlignments(self):
    self._testPharaohAlignments("", [0, 0], [])
    self._testPharaohAlignments("0-0", [1, 1], [[1]])
    self._testPharaohAlignments(
        "0-0 1-1 2-2 3-3", [4, 4], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    self._testPharaohAlignments(
        "0-0 1-1 2-3 3-2", [4, 4], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    self._testPharaohAlignments(
        "0-0 1-2 1-1", [2, 3], [[1, 0], [0, 1], [0, 1]])
    self._testPharaohAlignments(
        "0-0 1-2 1-1 2-4", [3, 5], [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]])


if __name__ == "__main__":
  tf.test.main()
