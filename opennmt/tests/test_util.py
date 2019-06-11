import unittest

from opennmt import constants
from opennmt.data import vocab
from opennmt.utils import compat


def skip_if_unsupported(symbol):
  return unittest.skipIf(not compat.tf_supports(symbol), "tf.%s is not supported")

def make_data_file(path, lines):
  with open(path, "w") as data:
    for line in lines:
      data.write("%s\n" % line)
  return path

def make_vocab_from_file(path, data_file):
  vocabulary = vocab.Vocab(special_tokens=[
      constants.PADDING_TOKEN,
      constants.START_OF_SENTENCE_TOKEN,
      constants.END_OF_SENTENCE_TOKEN])
  vocabulary.add_from_text(data_file)
  vocabulary.serialize(path)
  return path
