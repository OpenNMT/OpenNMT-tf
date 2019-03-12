# -*- coding: utf-8 -*-

import os
import six
import yaml

import tensorflow as tf
import numpy as np

try:
  from tensorflow.contrib.tensorboard.plugins import projector
except ModuleNotFoundError:
  from tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt import tokenizers
from opennmt.constants import PADDING_TOKEN as PAD
from opennmt.inputters import inputter, text_inputter, record_inputter
from opennmt.layers import reducer
from opennmt.utils import compat, data
from opennmt.utils.misc import item_or_tuple, count_lines
from opennmt.tests import test_util


class InputterTest(tf.test.TestCase):

  @test_util.run_tf1_only
  def testVisualizeEmbeddings(self):
    log_dir = os.path.join(self.get_temp_dir(), "log")
    if not os.path.exists(log_dir):
      os.mkdir(log_dir)

    def _create_embedding(name, vocab_filename, vocab_size=10, num_oov_buckets=1):
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
      vocab_file = os.path.join(log_dir, "%s.txt" % embedding.op.name)
      self.assertTrue(os.path.exists(projector_config_path))
      self.assertTrue(os.path.exists(vocab_file))
      self.assertEqual(embedding.get_shape().as_list()[0], count_lines(vocab_file))
      with open(projector_config_path) as projector_config_file:
        text_format.Merge(projector_config_file.read(), projector_config)
      return projector_config

    # Register an embedding variable.
    src_embedding, src_vocab_file = _create_embedding("src_emb", "src_vocab.txt")
    projector_config = _visualize(src_embedding, src_vocab_file)
    self.assertEqual(1, len(projector_config.embeddings))
    self.assertEqual(src_embedding.name, projector_config.embeddings[0].tensor_name)
    self.assertEqual("src_emb.txt", projector_config.embeddings[0].metadata_path)

    # Register a second embedding variable.
    tgt_embedding, tgt_vocab_file = _create_embedding(
        "tgt_emb", "tgt_vocab.txt", num_oov_buckets=2)
    projector_config = _visualize(tgt_embedding, tgt_vocab_file, num_oov_buckets=2)
    self.assertEqual(2, len(projector_config.embeddings))
    self.assertEqual(tgt_embedding.name, projector_config.embeddings[1].tensor_name)
    self.assertEqual("tgt_emb.txt", projector_config.embeddings[1].metadata_path)

    # Update an existing variable.
    tf.reset_default_graph()
    src_embedding, src_vocab_file = _create_embedding("src_emb", "src_vocab.txt", vocab_size=20)
    projector_config = _visualize(src_embedding, src_vocab_file)
    self.assertEqual(2, len(projector_config.embeddings))
    self.assertEqual(src_embedding.name, projector_config.embeddings[0].tensor_name)
    self.assertEqual("src_emb.txt", projector_config.embeddings[0].metadata_path)

  def _testTokensToChars(self, tokens, expected_chars, expected_lengths):
    expected_chars = compat.nest.map_structure(tf.compat.as_bytes, expected_chars)
    chars, lengths = text_inputter.tokens_to_chars(tf.constant(tokens, dtype=tf.string))
    chars, lengths = self.evaluate([chars, lengths])
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

  def _makeTextFile(self, name, lines):
    path = os.path.join(self.get_temp_dir(), name)
    with open(path, "w") as f:
      for line in lines:
        f.write("%s\n" % line)
    return path

  def _makeEmbeddingsFile(self, vectors, name="embedding", header=False):
    path = os.path.join(self.get_temp_dir(), name)
    with open(path, "w") as embs:
      if header:
        embs.write("%d %d\n" % (len(vectors), len(vectors[0][1])))
      for word, vector in vectors:
        embs.write("%s %s\n" % (word, " ".join(str(v) for v in vector)))
    return path

  def testPretrainedEmbeddingsLoading(self):
    vocab_file = self._makeTextFile("vocab.txt", ["Toto", "tOTO", "tata", "tete"])
    embedding_file = self._makeEmbeddingsFile(
        [("toto", [1, 1]), ("titi", [2, 2]), ("tata", [3, 3])])

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
    vocab_file = self._makeTextFile("vocab.txt", ["Toto", "tOTO", "tata", "tete"])
    embedding_file = self._makeEmbeddingsFile(
        [("toto", [1, 1]), ("titi", [2, 2]), ("tata", [3, 3])], header=True)

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

    if compat.is_tf2():
      iterator = None
      features = iter(dataset).next()
    else:
      iterator = dataset.make_initializable_iterator()
      features = iterator.get_next()

    if shapes is not None:
      all_features = [features]
      if not compat.is_tf2() and not inputter.is_target:
        all_features.append(inputter.get_serving_input_receiver().features)
      for f in all_features:
        for field, shape in six.iteritems(shapes):
          self.assertIn(field, f)
          self.assertTrue(f[field].shape.is_compatible_with(shape))

    inputs = inputter.make_inputs(features, training=True)
    if not compat.is_tf2():
      with self.test_session() as sess:
        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(iterator.initializer)
    return self.evaluate((features, inputs))

  def testWordEmbedder(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    embedder = text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={"tokens": [None, None], "ids": [None, None], "length": [None]})

    self.assertEqual(embedder.embedding.name, "w_embs:0")
    self.assertAllEqual([3], features["length"])
    self.assertAllEqual([[2, 1, 4]], features["ids"])
    self.assertAllEqual([1, 3, 10], transformed.shape)

  def testWordEmbedderTarget(self):
    vocab_file = self._makeTextFile(
        "vocab.txt", ["<blank>", "<s>", "</s>", "the", "world", "hello", "toto"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    embedder = text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)
    embedder.is_target = True
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={
            "tokens": [None, None],
            "ids": [None, None],
            "ids_out": [None, None],
            "length": [None]
        })

    self.assertAllEqual([4], features["length"])
    self.assertAllEqual([[1, 5, 4, 7]], features["ids"])
    self.assertAllEqual([[5, 4, 7, 2]], features["ids_out"])

  def testWordEmbedderWithTokenizer(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "￭"])
    data_file = self._makeTextFile("data.txt", ["hello world!"])
    tokenization = {
        "mode": "aggressive",
        "joiner_annotate": True,
        "joiner_new": True
    }
    tokenization_config_path = os.path.join(self.get_temp_dir(), "tok.yml")
    with open(tokenization_config_path, "w") as tokenization_config_file:
      yaml.dump(tokenization, tokenization_config_file)

    embedder = text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)
    metadata = {
        "vocabulary_file": vocab_file,
        "tokenization": tokenization_config_path
    }
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata=metadata,
        shapes={"tokens": [None, None], "ids": [None, None], "length": [None]})

    self.assertAllEqual([4], features["length"])
    self.assertAllEqual([[2, 1, 3, 4]], features["ids"])

  def testWordEmbedderWithPretrainedEmbeddings(self):
    data_file = self._makeTextFile("data.txt", ["hello world !"])
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    embedding_file = self._makeEmbeddingsFile(
        [("hello", [1, 1]), ("world", [2, 2]), ("toto", [3, 3])])

    embedder = text_inputter.WordEmbedder(
        "vocabulary_file",
        embedding_file_key="embedding_file",
        embedding_file_with_header=False)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file, "embedding_file": embedding_file},
        shapes={"tokens": [None, None], "ids": [None, None], "length": [None]})

    self.assertAllEqual([1, 1], transformed[0][0])
    self.assertAllEqual([2, 2], transformed[0][1])

  def testWordEmbedderWithPretrainedEmbeddingsInInitialize(self):
    data_file = self._makeTextFile("data.txt", ["hello world !"])
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    embedding_file = self._makeEmbeddingsFile(
        [("hello", [1, 1]), ("world", [2, 2]), ("toto", [3, 3])])

    embedder = text_inputter.WordEmbedder("vocabulary_file")
    metadata = {
        "vocabulary_file": vocab_file,
        "embedding": {
            "path": embedding_file,
            "with_header": False
        }
    }
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata=metadata,
        shapes={"tokens": [None, None], "ids": [None, None], "length": [None]})

    self.assertAllEqual([1, 1], transformed[0][0])
    self.assertAllEqual([2, 2], transformed[0][1])

  @test_util.run_tf1_only
  def testCharConvEmbedder(self):
    vocab_file = self._makeTextFile("vocab.txt", ["h", "e", "l", "w", "o"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    embedder = text_inputter.CharConvEmbedder("vocabulary_file", 10, 5)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={"char_ids": [None, None, None], "length": [None]})

    self.assertAllEqual([3], features["length"])
    self.assertAllEqual(
        [[[0, 1, 2, 2, 4], [3, 4, 5, 2, 5], [5, 5, 5, 5, 5]]],
        features["char_ids"])
    self.assertAllEqual([1, 3, 5], transformed.shape)

  @test_util.run_tf1_only
  def testCharRNNEmbedder(self):
    vocab_file = self._makeTextFile("vocab.txt", ["h", "e", "l", "w", "o"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    embedder = text_inputter.CharRNNEmbedder("vocabulary_file", 10, 5)
    features, transformed = self._makeDataset(
        embedder,
        data_file,
        metadata={"vocabulary_file": vocab_file},
        shapes={"char_ids": [None, None, None], "length": [None]})

    self.assertAllEqual([1, 3, 5], transformed.shape)

  def testParallelInputter(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    data_files = [data_file, data_file]

    parallel_inputter = inputter.ParallelInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.WordEmbedder("vocabulary_file_2", embedding_size=5)])
    self.assertEqual(parallel_inputter.num_outputs, 2)
    features, transformed = self._makeDataset(
        parallel_inputter,
        data_files,
        metadata={"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_file},
        shapes={"inputter_0_ids": [None, None], "inputter_0_length": [None],
                "inputter_1_ids": [None, None], "inputter_1_length": [None]})

    self.assertEqual(2, len(parallel_inputter.get_length(features)))
    self.assertEqual(2, len(transformed))
    self.assertAllEqual([1, 3, 10], transformed[0].shape)
    self.assertAllEqual([1, 3, 5], transformed[1].shape)

  def testParallelInputterShareParameters(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    metadata = {"vocabulary_file": vocab_file}
    inputters = [
        text_inputter.WordEmbedder("vocabulary_file", embedding_size=10),
        text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)]
    parallel_inputter = inputter.ParallelInputter(inputters, share_parameters=True)
    parallel_inputter.initialize(metadata)
    parallel_inputter.build()
    self.assertEqual(inputters[0].embedding, inputters[1].embedding)

  def testNestedParallelInputterShareParameters(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    metadata = {"vocabulary_file": vocab_file}
    source_inputters = [
        text_inputter.WordEmbedder("vocabulary_file", embedding_size=10),
        text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)]
    target_inputter = text_inputter.WordEmbedder("vocabulary_file", embedding_size=10)
    inputters = [
        inputter.ParallelInputter(source_inputters, share_parameters=True),
        target_inputter]
    parallel_inputter = inputter.ParallelInputter(inputters, share_parameters=True)
    parallel_inputter.initialize(metadata)
    parallel_inputter.build()
    self.assertEqual(source_inputters[0].embedding, target_inputter.embedding)
    self.assertEqual(source_inputters[1].embedding, target_inputter.embedding)

  def testExampleInputter(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    source_inputter = text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10)
    target_inputter = text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10)
    example_inputter = inputter.ExampleInputter(source_inputter, target_inputter)
    self.assertEqual(example_inputter.num_outputs, 2)

    features, transformed = self._makeDataset(
        example_inputter,
        [data_file, data_file],
        metadata={"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_file})

    self.assertIsInstance(features, tuple)
    self.assertEqual(len(features), 2)
    self.assertEqual(len(transformed), 2)
    features, labels = features
    for field in ("ids", "length", "tokens"):
      self.assertIn(field, features)
    for field in ("ids", "ids_out", "length", "tokens"):
      self.assertIn(field, labels)

  def testExampleInputterAsset(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    source_inputter = text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10)
    target_inputter = text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10)
    example_inputter = inputter.ExampleInputter(source_inputter, target_inputter)
    example_inputter.initialize({
        "vocabulary_file_1": vocab_file,
        "vocabulary_file_2": vocab_file,
        "source_tokenization": {"mode": "conservative"}
    })
    self.assertIsInstance(source_inputter.tokenizer, tokenizers.OpenNMTTokenizer)

  @test_util.run_tf1_only
  def testMixedInputter(self):
    vocab_file = self._makeTextFile("vocab.txt", ["the", "world", "hello", "toto"])
    vocab_alt_file = self._makeTextFile("vocab_alt.txt", ["h", "e", "l", "w", "o"])
    data_file = self._makeTextFile("data.txt", ["hello world !"])

    mixed_inputter = inputter.MixedInputter([
        text_inputter.WordEmbedder("vocabulary_file_1", embedding_size=10),
        text_inputter.CharConvEmbedder("vocabulary_file_2", 10, 5)],
        reducer=reducer.ConcatReducer())
    self.assertEqual(mixed_inputter.num_outputs, 1)
    features, transformed = self._makeDataset(
        mixed_inputter,
        data_file,
        metadata={"vocabulary_file_1": vocab_file, "vocabulary_file_2": vocab_alt_file},
        shapes={"char_ids": [None, None, None], "ids": [None, None], "length": [None]})
    self.assertAllEqual([1, 3, 15], transformed.shape)

  def testSequenceRecord(self):
    vector = np.array([[0.2, 0.3], [0.4, 0.5]], dtype=np.float32)

    record_file = os.path.join(self.get_temp_dir(), "data.records")
    writer = compat.tf_compat(v2="io.TFRecordWriter", v1="python_io.TFRecordWriter")(record_file)
    record_inputter.write_sequence_record(vector, writer)
    writer.close()

    inputter = record_inputter.SequenceRecordInputter()
    features, transformed = self._makeDataset(
        inputter,
        record_file,
        shapes={"tensor": [None, None, 2], "length": [None]})

    self.assertEqual([2], features["length"])
    self.assertAllEqual([vector], features["tensor"])
    self.assertAllEqual([vector], transformed)


if __name__ == "__main__":
  tf.test.main()
