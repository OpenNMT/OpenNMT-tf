import os

import tensorflow as tf

from opennmt.utils import Vocab


class VocabTest(tf.test.TestCase):

  def testSimpleVocab(self):
    vocab = Vocab()

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
    vocab = Vocab(special_tokens=["foo", "bar"])

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
    vocab1 = Vocab(special_tokens=["foo", "bar"])
    vocab1.add("toto")
    vocab1.add("toto")
    vocab1.add("toto")
    vocab1.add("titi")
    vocab1.add("titi")
    vocab1.add("tata")

    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")

    vocab1.serialize(vocab_file)
    vocab2 = Vocab()
    vocab2.add_from_text(vocab_file)

    self.assertEqual(vocab1.size, vocab2.size)
    self.assertEqual(vocab1.lookup("titi"), vocab2.lookup("titi"))


if __name__ == "__main__":
  tf.test.main()
