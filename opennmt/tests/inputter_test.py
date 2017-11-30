# -*- coding: utf-8 -*-

import os

import tensorflow as tf

from opennmt.constants import PADDING_TOKEN as PAD
from opennmt.inputters import inputter, text_inputter
from opennmt.utils import reducer


embedding_file = "inputter_test_embedding.tmp"
vocab_file = "inputter_test_vocab.tmp"
vocab_alt_file = "inputter_test_vocab_alt.tmp"
data_file = "inputter_test_data.tmp"


def _first_element(inputter, data_file, metadata):
  inputter.initialize(metadata)
  dataset = inputter.make_dataset(data_file)
  iterator = dataset.make_initializable_iterator()
  tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
  next_element = iterator.get_next()
  data = inputter.process(next_element)
  for key, value in data.items():
    data[key] = tf.expand_dims(value, 0)
  transformed = inputter.transform_data(data)
  return data, transformed


class InputterTest(tf.test.TestCase):

  def tearDown(self):
    if os.path.isfile(embedding_file):
      os.remove(embedding_file)
    if os.path.isfile(vocab_file):
      os.remove(vocab_file)
    if os.path.isfile(vocab_alt_file):
      os.remove(vocab_alt_file)
    if os.path.isfile(data_file):
      os.remove(data_file)


  def _testTokensToChars(self, tokens, expected_chars, expected_lengths):
    expected_chars = [[tf.compat.as_bytes(c) for c in w] for w in expected_chars]
    chars, lengths = text_inputter.tokens_to_chars(tf.constant(tokens))
    with self.test_session() as sess:
      chars, lengths = sess.run([chars, lengths])
      self.assertAllEqual(expected_chars, chars)
      self.assertAllEqual(expected_lengths, lengths)

  def testTokensToCharsSingle(self):
    self._testTokensToChars(["Hello"], [["H", "e", "l", "l", "o"]], [5])

  def testTokensToCharsMixed(self):
    self._testTokensToChars(
        ["Just", "a", "测试"],
        [["J", "u", "s", "t"], ["a", PAD, PAD, PAD], ["测", "试", PAD, PAD]],
        [4, 1, 2])

  def testPretrainedEmbeddingsLoading(self):
    with open(embedding_file, "w") as embedding:
      embedding.write("toto 1 1\n"
                      "titi 2 2\n"
                      "tata 3 3\n")
    with open(vocab_file, "w") as vocab:
      vocab.write("Toto\n"
                  "tOTO\n"
                  "tata\n"
                  "tete\n")

    embeddings = text_inputter.load_pretrained_embeddings(
        embedding_file,
        vocab_file,
        num_oov_buckets=1,
        with_header=False,
        case_insensitive_embeddings=True)
    self.assertAllEqual([5, 2], embeddings.shape)
    self.assertAllEqual([1, 1], embeddings[0])
    self.assertAllEqual([1, 1], embeddings[1])
    self.assertAllEqual([3, 3], embeddings[2])

    embeddings = text_inputter.load_pretrained_embeddings(
        embedding_file,
        vocab_file,
        num_oov_buckets=2,
        with_header=False,
        case_insensitive_embeddings=False)
    self.assertAllEqual([6, 2], embeddings.shape)
    self.assertAllEqual([3, 3], embeddings[2])

  def testPretrainedEmbeddingsWithHeaderLoading(self):
    with open(embedding_file, "w") as embedding:
      embedding.write("3 2\n"
                      "toto 1 1\n"
                      "titi 2 2\n"
                      "tata 3 3\n")
    with open(vocab_file, "w") as vocab:
      vocab.write("Toto\n"
                  "tOTO\n"
                  "tata\n"
                  "tete\n")

    embeddings = text_inputter.load_pretrained_embeddings(
        embedding_file,
        vocab_file,
        num_oov_buckets=1,
        case_insensitive_embeddings=True)
    self.assertAllEqual([5, 2], embeddings.shape)
    self.assertAllEqual([1, 1], embeddings[0])
    self.assertAllEqual([1, 1], embeddings[1])
    self.assertAllEqual([3, 3], embeddings[2])

  def testWordEmbedder(self):
    with open(vocab_file, "w") as vocab:
      vocab.write("the\n"
                  "world\n"
                  "hello\n"
                  "toto\n")
    with open(data_file, "w") as data:
      data.write("hello world !\n")

    embedder = text_inputter.WordEmbedder(
        "vocabulary_file", embedding_size=10)
    data, transformed = _first_element(
        embedder, data_file, {"vocabulary_file": vocab_file})

    input_receiver = embedder.get_serving_input_receiver()
    self.assertAllEqual(
        [None, None],
        input_receiver.features["ids"].get_shape().as_list())
    self.assertAllEqual(
        [None],
        input_receiver.features["length"].get_shape().as_list())

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      data, transformed = sess.run([data, transformed])
      self.assertNotIn("raw", data)
      self.assertNotIn("tokens", data)
      self.assertAllEqual([3], data["length"])
      self.assertAllEqual([[2, 1, 4]], data["ids"])
      self.assertAllEqual([1, 3, 10], transformed.shape)

  def testCharConvEmbedder(self):
    with open(vocab_file, "w") as vocab:
      vocab.write("h\n"
                  "e\n"
                  "l\n"
                  "w\n"
                  "o\n")
    with open(data_file, "w") as data:
      data.write("hello world !\n")

    embedder = text_inputter.CharConvEmbedder("vocabulary_file", 10, 5)
    data, transformed = _first_element(
        embedder, data_file, {"vocabulary_file": vocab_file})

    input_receiver = embedder.get_serving_input_receiver()
    self.assertAllEqual(
        [None, None, None],
        input_receiver.features["char_ids"].get_shape().as_list())
    self.assertAllEqual(
        [None],
        input_receiver.features["length"].get_shape().as_list())

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      data, transformed = sess.run([data, transformed])
      self.assertNotIn("raw", data)
      self.assertNotIn("tokens", data)
      self.assertAllEqual([3], data["length"])
      self.assertAllEqual(
          [[[0, 1, 2, 2, 4], [3, 4, 5, 2, 5], [5, 5, 5, 5, 5]]],
          data["char_ids"])
      self.assertAllEqual([1, 3, 5], transformed.shape)

  def testParallelInputter(self):
    with open(vocab_file, "w") as vocab:
      vocab.write("the\n"
                  "world\n"
                  "hello\n"
                  "toto\n")
    with open(data_file, "w") as data:
      data.write("hello world !\n")

    parallel_inputter = inputter.ParallelInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.WordEmbedder("vocabulary_file_2", embedding_size=5)])

    data, transformed = _first_element(
        parallel_inputter,
        [data_file, data_file],
        {"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_file})

    self.assertEqual(2, len(parallel_inputter.get_length(data)))

    input_receiver = parallel_inputter.get_serving_input_receiver()
    self.assertIn("inputter_0_ids", input_receiver.features)
    self.assertIn("inputter_1_ids", input_receiver.features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      data, transformed = sess.run([data, transformed])
      self.assertNotIn("inputter_0_raw", data)
      self.assertNotIn("inputter_0_tokens", data)
      self.assertNotIn("inputter_1_raw", data)
      self.assertNotIn("inputter_1_tokens", data)
      self.assertIn("inputter_0_ids", data)
      self.assertIn("inputter_1_ids", data)
      self.assertEqual(2, len(transformed))
      self.assertAllEqual([1, 3, 10], transformed[0].shape)
      self.assertAllEqual([1, 3, 5], transformed[1].shape)

  def testMixedInputter(self):
    with open(vocab_file, "w") as vocab:
      vocab.write("the\n"
                  "world\n"
                  "hello\n"
                  "toto\n")
    with open(vocab_alt_file, "w") as vocab_alt:
      vocab_alt.write("h\n"
                      "e\n"
                      "l\n"
                      "w\n"
                      "o\n")
    with open(data_file, "w") as data:
      data.write("hello world !\n")

    mixed_inputter = inputter.MixedInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.CharConvEmbedder("vocabulary_file_2", 10, 5)],
        reducer=reducer.ConcatReducer())

    data, transformed = _first_element(
        mixed_inputter,
        data_file,
        {"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_alt_file})

    input_receiver = mixed_inputter.get_serving_input_receiver()
    self.assertIn("ids", input_receiver.features)
    self.assertIn("char_ids", input_receiver.features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      data, transformed = sess.run([data, transformed])
      self.assertNotIn("raw", data)
      self.assertNotIn("tokens", data)
      self.assertIn("ids", data)
      self.assertIn("char_ids", data)
      self.assertAllEqual([1, 3, 15], transformed.shape)


if __name__ == "__main__":
  tf.test.main()
