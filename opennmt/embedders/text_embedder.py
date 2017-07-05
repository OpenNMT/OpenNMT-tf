"""Define word-based embedders."""

import os
import shutil

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

  # If the projectpr file exists, load it.
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
               embedding_size,
               dropout_keep_prob=1.0,
               name=None):
    """Initializes the parameters of the word embedder.

    Args:
      vocabulary_file: The vocabulary filename.
      embedding_size: The size of the resulting embedding.
      dropout_keep_prob: The probability to keep units in the embedding.
      name: The name of this embedders used to prefix data fields.
    """
    super(WordEmbedder, self).__init__(name=name)

    self.vocabulary_file = vocabulary_file
    self.embedding_size = embedding_size
    self.dropout_keep_prob = dropout_keep_prob

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
    embeddings = tf.get_variable(
      "w_embs", shape=[self.vocabulary_size, self.embedding_size])

    outputs = tf.nn.embedding_lookup(embeddings, inputs)

    if mode == tf.estimator.ModeKeys.TRAIN and self.dropout_keep_prob < 1.0:
      outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)

    return outputs
