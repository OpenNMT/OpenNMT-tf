import os

import tensorflow as tf
import numpy as np

from opennmt.utils import checkpoint
from opennmt.utils.vocab import Vocab


class CheckpointTest(tf.test.TestCase):

  def _saveVocab(self, name, words):
    vocab = Vocab()
    for word in words:
      vocab.add(str(word))
    vocab_file = os.path.join(self.get_temp_dir(), name)
    vocab.serialize(vocab_file)
    return vocab_file

  def testVocabMappingMerge(self):
    old = self._saveVocab("old", [1, 2, 3, 4])
    new = self._saveVocab("new", [1, 6, 3, 5, 7])
    mapping = checkpoint._get_vocabulary_mapping(old, new, "merge")
    self.assertEqual(4 + 5 - 2 + 1, len(mapping))  # old + new - common + <unk>
    self.assertAllEqual([0, 1, 2, 3, -1, -1, -1, 4], mapping)

  def testVocabMappingReplace(self):
    old = self._saveVocab("old", [1, 2, 3, 4])
    new = self._saveVocab("new", [1, 6, 5, 3, 7])
    mapping = checkpoint._get_vocabulary_mapping(old, new, "replace")
    self.assertEqual(5 + 1, len(mapping))  # new + <unk>
    self.assertAllEqual([0, -1, -1, 2, -1, 4], mapping)

  def testVocabVariableUpdate(self):
    mapping = [0, -1, -1, 2, -1, 4]
    old = np.array([1, 2, 3, 4, 5, 6, 7])
    vocab_size = 7
    new = checkpoint._update_vocabulary_variable(old, vocab_size, mapping)
    self.assertAllEqual([1, 0, 0, 3, 0, 5], new)


if __name__ == "__main__":
  tf.test.main()
