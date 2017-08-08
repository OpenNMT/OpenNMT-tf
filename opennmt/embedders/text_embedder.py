"""Define word-based embedders."""

import os
import shutil

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt.embedders.embedder import Embedder
from opennmt.utils.misc import count_lines


def _visualize(log_dir, embedding_var, vocabulary_file):
  # Copy vocabulary file to log_dir.
  if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

  basename = os.path.basename(vocabulary_file)
  destination = os.path.join(log_dir, basename)
  shutil.copy(vocabulary_file, destination)

  config = projector.ProjectorConfig()

  # If the projector file exists, load it.
  target = os.path.join(log_dir, "projector_config.pbtxt")
  if os.path.exists(target):
    with open(target) as target_file:
      text_format.Merge(target_file.read(), config)

  # If this embedding is already registered, just update the metadata path.
  exists = False
  for meta in config.embeddings:
    if meta.tensor_name == embedding_var.name:
      meta.metadata_path = basename
      exists = True
      break

  if not exists:
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = basename

  summary_writer = tf.summary.FileWriter(log_dir)

  projector.visualize_embeddings(summary_writer, config)

def _load_pretrained_embeddings(embedding_file, vocabulary_file):
  # Map words to ids from the vocabulary.
  word_to_id = {}
  with open(vocabulary_file) as vocabulary:
    count = 0
    for word in vocabulary:
      word_to_id[word.strip()] = count
      count += 1

  # Fill pretrained embedding matrix.
  with open(embedding_file) as embedding:
    pretrained = None

    for line in embedding:
      fields = line.strip().split()
      word = fields[0]

      if pretrained is None:
        pretrained = np.random.normal(size=(count + 1, len(fields) - 1))

      # Lookup word in the vocabulary.
      if word in word_to_id:
        index = word_to_id[word]
        pretrained[index] = np.asarray(fields[1:])

  return pretrained


class TextEmbedder(Embedder):
  """An abstract embedder that process text."""

  def __init__(self, name=None):
    super(TextEmbedder, self).__init__(name=name)

  def process(self, data):
    """Tokenizes raw text."""
    data = super(TextEmbedder, self).process(data)

    text = self.get_data_field(data, "raw")
    tokens = tf.string_split([text]).values
    length = tf.shape(tokens)[0]

    data = self.set_data_field(data, "tokens", tokens, padded_shape=[None])
    data = self.set_data_field(data, "length", length, padded_shape=[])

    return data


class WordEmbedder(TextEmbedder):
  """Simple word embedder."""

  def __init__(self,
               vocabulary_file,
               embedding_size=None,
               embedding_file=None,
               trainable=True,
               dropout=0.0,
               name=None):
    """Initializes the parameters of the word embedder.

    Args:
      vocabulary_file: The vocabulary file containing one word per line.
      embedding_size: The size of the resulting embedding.
        If `None`, an embedding file must be provided.
      embedding_file: The embedding file with the format:
        ```
        word1 val1 val2 ... valM
        word2 val1 val2 ... valM
        ...
        wordN val1 val2 ... valM
        ```
        Entries will be matched against `vocabulary_file`.
      trainable: If `False`, do not optimize embeddings.
      dropout: The probability to drop units in the embedding.
      name: The name of this embedders used to prefix data fields.
    """
    super(WordEmbedder, self).__init__(name=name)

    self.vocabulary_file = vocabulary_file
    self.embedding_size = embedding_size
    self.embedding_file = embedding_file
    self.trainable = trainable
    self.dropout = dropout

    if embedding_size is None and embedding_file is None:
      raise ValueError("Must either provide embedding_size or embedding_file")

    self.num_oov_buckets = 1
    self.vocabulary_size = count_lines(vocabulary_file) + self.num_oov_buckets

  def init(self):
    self.vocabulary = tf.contrib.lookup.index_table_from_file(
      self.vocabulary_file,
      vocab_size=self.vocabulary_size - self.num_oov_buckets,
      num_oov_buckets=self.num_oov_buckets)

  def process(self, data):
    """Converts words tokens to ids."""
    data = super(WordEmbedder, self).process(data)

    tokens = self.get_data_field(data, "tokens")
    ids = self.vocabulary.lookup(tokens)

    data = self.set_data_field(data, "ids", ids, padded_shape=[None])
    return data

  def visualize(self, log_dir):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      embeddings = tf.get_variable(
        "w_embs", shape=[self.vocabulary_size, self.embedding_size])

      _visualize(log_dir, embeddings, self.vocabulary_file)

  def embed_from_data(self, data, mode):
    return self.embed(self.get_data_field(data, "ids"), mode)

  def _embed(self, inputs, mode):
    try:
      embeddings = tf.get_variable("w_embs", trainable=self.trainable)
    except ValueError:
      # Variable does not exist yet.
      if self.embedding_file:
        pretrained = _load_pretrained_embeddings(self.embedding_file, self.vocabulary_file)
        self.embedding_size = pretrained.shape[-1]

        shape = None
        initializer = tf.constant(pretrained.astype(np.float32))
      else:
        shape = [self.vocabulary_size, self.embedding_size]
        initializer = None

      embeddings = tf.get_variable(
        "w_embs",
        shape=shape,
        initializer=initializer,
        trainable=self.trainable)

    outputs = tf.nn.embedding_lookup(embeddings, inputs)

    outputs = tf.contrib.layers.dropout(
      outputs,
      keep_prob=1.0 - self.dropout,
      is_training=mode == tf.estimator.ModeKeys.TRAIN)

    return outputs
