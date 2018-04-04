# -*- coding: utf-8 -*-

import io
import os
import six

import tensorflow as tf
import numpy as np

from tensorflow.contrib.tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt.constants import PADDING_TOKEN as PAD
from opennmt.inputters import inputter, text_inputter, record_inputter
from opennmt.layers import reducer
from opennmt.utils import data
from opennmt.utils.misc import item_or_tuple, count_lines


class InputterTest(tf.test.TestCase):

  def testVisualizeEmbeddings(self):
    log_dir = os.path.join(self.get_temp_dir(), "log")
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)

    def _create_embedding(name, vocab_filename, num_oov_buckets=1):
      vocab_size = 10
      vocab_file = os.path.join(self.get_temp_dir(), vocab_filename)
      with open(vocab_file, mode="wb") as vocab:
        for i in range(vocab_size):
          vocab.write(tf.compat.as_bytes("%d\n" % i))
      variable = tf.get_variable(name, shape=[vocab_size + num_oov_buckets, 4])
      return variable, vocab_file

    def _visualize(embedding, vocab_file, num_oov_buckets=1):
      text_inputter.visualize_embeddings(
          log_dir, embedding, vocab_file, num_oov_buckets=num_oov_buckets)
      projector_config = projector.ProjectorConfig()
      projector_config_path = os.path.join(log_dir, "projector_config.pbtxt")
      vocab_file = os.path.join(log_dir, os.path.basename(vocab_file))
      self.assertTrue(os.path.exists(projector_config_path))
      self.assertTrue(os.path.exists(vocab_file))
      self.assertEqual(embedding.get_shape().as_list()[0], count_lines(vocab_file))
      with io.open(projector_config_path, encoding="utf-8") as projector_config_file:
        text_format.Merge(projector_config_file.read(), projector_config)
      return projector_config

    # Register an embedding variable.
    src_embedding, src_vocab_file = _create_embedding("src_emb", "src_vocab.txt")
    projector_config = _visualize(src_embedding, src_vocab_file)
    self.assertEqual(1, len(projector_config.embeddings))
    self.assertEqual(src_embedding.name, projector_config.embeddings[0].tensor_name)
    self.assertEqual("src_vocab.txt", projector_config.embeddings[0].metadata_path)

    # Register a second embedding variable.
    tgt_embedding, tgt_vocab_file = _create_embedding(
        "tgt_emb", "tgt_vocab.txt", num_oov_buckets=2)
    projector_config = _visualize(tgt_embedding, tgt_vocab_file, num_oov_buckets=2)
    self.assertEqual(2, len(projector_config.embeddings))
    self.assertEqual(tgt_embedding.name, projector_config.embeddings[1].tensor_name)
    self.assertEqual("tgt_vocab.txt", projector_config.embeddings[1].metadata_path)

    # Update an existing embedding variable with another vocabulary path.
    new_src_vocab_file = os.path.join(self.get_temp_dir(), "src_vocab_new.txt")
    os.rename(src_vocab_file, new_src_vocab_file)
    projector_config = _visualize(src_embedding, new_src_vocab_file)
    self.assertEqual(2, len(projector_config.embeddings))
    self.assertEqual(src_embedding.name, projector_config.embeddings[0].tensor_name)
    self.assertEqual("src_vocab_new.txt", projector_config.embeddings[0].metadata_path)

  def _testTokensToChars(self, tokens, expected_chars, expected_lengths):
    expected_chars = [[tf.compat.as_bytes(c) for c in w] for w in expected_chars]
    tokens = tf.placeholder_with_default(tokens, shape=[None])
    chars, lengths = text_inputter.tokens_to_chars(tokens)
    with self.test_session() as sess:
      chars, lengths = sess.run([chars, lengths])
      self.assertListEqual(expected_chars, chars.tolist())
      self.assertListEqual(expected_lengths, lengths.tolist())

  def testTokensToCharsEmpty(self):
    self._testTokensToChars([], [], [])

  def testTokensToCharsSingle(self):
    self._testTokensToChars(["Hello"], [["H", "e", "l", "l", "o"]], [5])

  def testTokensToCharsMixed(self):
    self._testTokensToChars(
        ["Just", "a", "测试"],
        [["J", "u", "s", "t"], ["a", PAD, PAD, PAD], ["测", "试", PAD, PAD]],
        [4, 1, 2])

  def testPretrainedEmbeddingsLoading(self):
    embedding_file = os.path.join(self.get_temp_dir(), "embedding.txt")
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")

    with io.open(embedding_file, encoding="utf-8", mode="w") as embedding:
      embedding.write(u"toto 1 1\n"
                      u"titi 2 2\n"
                      u"tata 3 3\n")
    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"Toto\n"
                  u"tOTO\n"
                  u"tata\n"
                  u"tete\n")

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
    embedding_file = os.path.join(self.get_temp_dir(), "embedding.txt")
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")

    with io.open(embedding_file, encoding="utf-8", mode="w") as embedding:
      embedding.write(u"3 2\n"
                      u"toto 1 1\n"
                      u"titi 2 2\n"
                      u"tata 3 3\n")
    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"Toto\n"
                  u"tOTO\n"
                  u"tata\n"
                  u"tete\n")

    embeddings = text_inputter.load_pretrained_embeddings(
        embedding_file,
        vocab_file,
        num_oov_buckets=1,
        case_insensitive_embeddings=True)
    self.assertAllEqual([5, 2], embeddings.shape)
    self.assertAllEqual([1, 1], embeddings[0])
    self.assertAllEqual([1, 1], embeddings[1])
    self.assertAllEqual([3, 3], embeddings[2])

  def _makeDataset(self, inputter, data_file, metadata=None, dataset_size=1, shapes=None):
    if metadata is not None:
      inputter.initialize(metadata)

    self.assertEqual(dataset_size, inputter.get_dataset_size(data_file))

    dataset = inputter.make_dataset(data_file)
    dataset = dataset.map(lambda *arg: inputter.process(item_or_tuple(arg)))
    dataset = dataset.padded_batch(1, padded_shapes=data.get_padded_shapes(dataset))

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    next_element = iterator.get_next()

    if shapes is not None:
      for features in (next_element, inputter.get_serving_input_receiver().features):
        self.assertNotIn("raw", features)
        for field, shape in six.iteritems(shapes):
          self.assertIn(field, features)
          self.assertAllEqual(shape, features[field].get_shape().as_list())

    transformed = inputter.transform_data(next_element)
    return next_element, transformed

  def testWordEmbedder(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    data_file = os.path.join(self.get_temp_dir(), "data.txt")

    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"the\n"
                  u"world\n"
                  u"hello\n"
                  u"toto\n")
    with io.open(data_file, encoding="utf-8", mode="w") as data:
      data.write(u"hello world !\n")

    embedder = text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={"ids": [None, None], "length": [None]})

    self.assertNotIn("tokens", features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertAllEqual([3], features["length"])
      self.assertAllEqual([[2, 1, 4]], features["ids"])
      self.assertAllEqual([1, 3, 10], transformed.shape)

  def testWordEmbedderWithPretrainedEmbeddings(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    data_file = os.path.join(self.get_temp_dir(), "data.txt")
    embedding_file = os.path.join(self.get_temp_dir(), "embedding.txt")

    with io.open(embedding_file, encoding="utf-8", mode="w") as embedding:
      embedding.write(u"hello 1 1\n"
                      u"world 2 2\n"
                      u"toto 3 3\n")
    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"the\n"
                  u"world\n"
                  u"hello\n"
                  u"toto\n")
    with io.open(data_file, encoding="utf-8", mode="w") as data:
      data.write(u"hello world !\n")

    embedder = text_inputter.WordEmbedder(
        "vocabulary_file",
        embedding_file_key="embedding_file",
        embedding_file_with_header=False)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file, "embedding_file": embedding_file},
        shapes={"ids": [None, None], "length": [None]})

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertAllEqual([1, 1], transformed[0][0])
      self.assertAllEqual([2, 2], transformed[0][1])

  def testCharConvEmbedder(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    data_file = os.path.join(self.get_temp_dir(), "data.txt")

    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"h\n"
                  u"e\n"
                  u"l\n"
                  u"w\n"
                  u"o\n")
    with io.open(data_file, encoding="utf-8", mode="w") as data:
      data.write(u"hello world !\n")

    embedder = text_inputter.CharConvEmbedder("vocabulary_file", 10, 5)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={"char_ids": [None, None, None], "length": [None]})

    self.assertNotIn("tokens", features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertAllEqual([3], features["length"])
      self.assertAllEqual(
          [[[0, 1, 2, 2, 4], [3, 4, 5, 2, 5], [5, 5, 5, 5, 5]]],
          features["char_ids"])
      self.assertAllEqual([1, 3, 5], transformed.shape)

  def testParallelInputter(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    data_file = os.path.join(self.get_temp_dir(), "data.txt")

    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"the\n"
                  u"world\n"
                  u"hello\n"
                  u"toto\n")
    with io.open(data_file, encoding="utf-8", mode="w") as data:
      data.write(u"hello world !\n")

    data_files = [data_file, data_file]

    parallel_inputter = inputter.ParallelInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.WordEmbedder("vocabulary_file_2", embedding_size=5)])
    features, transformed = self._makeDataset(
        parallel_inputter,
        data_files,
        metadata={"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_file},
        shapes={"inputter_0_ids": [None, None], "inputter_0_length": [None],
                "inputter_1_ids": [None, None], "inputter_1_length": [None]})

    self.assertEqual(2, len(parallel_inputter.get_length(features)))
    self.assertNotIn("inputter_0_raw", features)
    self.assertNotIn("inputter_0_tokens", features)
    self.assertNotIn("inputter_1_raw", features)
    self.assertNotIn("inputter_1_tokens", features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertEqual(2, len(transformed))
      self.assertAllEqual([1, 3, 10], transformed[0].shape)
      self.assertAllEqual([1, 3, 5], transformed[1].shape)

  def testMixedInputter(self):
    vocab_file = os.path.join(self.get_temp_dir(), "vocab.txt")
    vocab_alt_file = os.path.join(self.get_temp_dir(), "vocab_alt.txt")
    data_file = os.path.join(self.get_temp_dir(), "data.txt")

    with io.open(vocab_file, encoding="utf-8", mode="w") as vocab:
      vocab.write(u"the\n"
                  u"world\n"
                  u"hello\n"
                  u"toto\n")
    with io.open(vocab_alt_file, encoding="utf-8", mode="w") as vocab_alt:
      vocab_alt.write(u"h\n"
                      u"e\n"
                      u"l\n"
                      u"w\n"
                      u"o\n")
    with io.open(data_file, encoding="utf-8", mode="w") as data:
      data.write(u"hello world !\n")

    mixed_inputter = inputter.MixedInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.CharConvEmbedder("vocabulary_file_2", 10, 5)],
        reducer=reducer.ConcatReducer())
    features, transformed = self._makeDataset(
        mixed_inputter,
        data_file,
        metadata={"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_alt_file},
        shapes={"char_ids": [None, None, None], "ids": [None, None], "length": [None]})

    self.assertNotIn("tokens", features)

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      sess.run(tf.global_variables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertAllEqual([1, 3, 15], transformed.shape)

  def testSequenceRecord(self):
    vector = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)

    record_file = os.path.join(self.get_temp_dir(), "data.records")
    writer = tf.python_io.TFRecordWriter(record_file)
    record_inputter.write_sequence_record(vector, writer)
    writer.close()

    inputter = record_inputter.SequenceRecordInputter()
    features, transformed = self._makeDataset(
        inputter,
        record_file,
        shapes={"tensor": [None, None, 2], "length": [None]})

    with self.test_session() as sess:
      sess.run(tf.tables_initializer())
      features, transformed = sess.run([features, transformed])
      self.assertEqual([2], features["length"])
      self.assertAllEqual([vector], features["tensor"])
      self.assertAllEqual([vector], transformed)


if __name__ == "__main__":
  tf.test.main()
