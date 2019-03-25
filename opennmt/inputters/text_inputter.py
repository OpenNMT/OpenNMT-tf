"""Define word-based embedders."""

import abc
import collections
import os
import six

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.lookup_ops import TextFileIndex
from tensorboard.plugins import projector
from google.protobuf import text_format

from opennmt import constants, tokenizers
from opennmt.inputters.inputter import Inputter
from opennmt.layers import common
from opennmt.utils.misc import count_lines
from opennmt.constants import PADDING_TOKEN


def visualize_embeddings(log_dir, embedding_var, vocabulary_file, num_oov_buckets=1):
  """Registers an embedding variable for visualization in TensorBoard.

  This function registers :obj:`embedding_var` in the ``projector_config.pbtxt``
  file and generates metadata from :obj:`vocabulary_file` to attach a label
  to each word ID.

  Args:
    log_dir: The active log directory.
    embedding_var: The embedding variable to visualize.
    vocabulary_file: The associated vocabulary file.
    num_oov_buckets: The number of additional unknown tokens.
  """
  # Copy vocabulary file to log_dir.
  basename = "%s.txt" % embedding_var.name[:-2].replace("/", "_")
  destination = os.path.join(log_dir, basename)
  tf.io.gfile.copy(vocabulary_file, destination, overwrite=True)

  # Append <unk> tokens.
  with tf.io.gfile.GFile(destination, mode="ab") as vocab:
    if num_oov_buckets == 1:
      vocab.write(b"<unk>\n")
    else:
      for i in range(num_oov_buckets):
        vocab.write(tf.compat.as_bytes("<unk%d>\n" % i))

  config = projector.ProjectorConfig()

  # If the projector file exists, load it.
  target = os.path.join(log_dir, "projector_config.pbtxt")
  if tf.io.gfile.exists(target):
    with tf.io.gfile.GFile(target, mode="rb") as target_file:
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

  summary_writer = tf.compat.v1.summary.FileWriter(log_dir)
  projector.visualize_embeddings(summary_writer, config)

def load_pretrained_embeddings(embedding_file,
                               vocabulary_file,
                               num_oov_buckets=1,
                               with_header=True,
                               case_insensitive_embeddings=True):
  """Returns pretrained embeddings relative to the vocabulary.

  The :obj:`embedding_file` must have the following format:

  .. code-block:: text

      N M
      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  or if :obj:`with_header` is ``False``:

  .. code-block:: text

      word1 val1 val2 ... valM
      word2 val1 val2 ... valM
      ...
      wordN val1 val2 ... valM

  This function will iterate on each embedding in :obj:`embedding_file` and
  assign the pretrained vector to the associated word in :obj:`vocabulary_file`
  if found. Otherwise, the embedding is ignored.

  If :obj:`case_insensitive_embeddings` is ``True``, word embeddings are assumed
  to be trained on lowercase data. In that case, word alignments are case
  insensitive meaning the pretrained word embedding for "the" will be assigned
  to "the", "The", "THE", or any other case variants included in
  :obj:`vocabulary_file`.

  Args:
    embedding_file: Path the embedding file. Entries will be matched against
      :obj:`vocabulary_file`.
    vocabulary_file: The vocabulary file containing one word per line.
    num_oov_buckets: The number of additional unknown tokens.
    with_header: ``True`` if the embedding file starts with a header line like
      in GloVe embedding files.
    case_insensitive_embeddings: ``True`` if embeddings are trained on lowercase
      data.

  Returns:
    A Numpy array of shape ``[vocabulary_size + num_oov_buckets, embedding_size]``.
  """
  # Map words to ids from the vocabulary.
  word_to_id = collections.defaultdict(list)
  with tf.io.gfile.GFile(vocabulary_file, mode="rb") as vocabulary:
    count = 0
    for word in vocabulary:
      word = word.strip()
      if case_insensitive_embeddings:
        word = word.lower()
      word_to_id[word].append(count)
      count += 1

  # Fill pretrained embedding matrix.
  with tf.io.gfile.GFile(embedding_file, mode="rb") as embedding:
    pretrained = None

    if with_header:
      next(embedding)

    for line in embedding:
      fields = line.strip().split()
      word = fields[0]

      if pretrained is None:
        pretrained = np.random.normal(
            size=(count + num_oov_buckets, len(fields) - 1))

      # Lookup word in the vocabulary.
      if word in word_to_id:
        ids = word_to_id[word]
        for index in ids:
          pretrained[index] = np.asarray(fields[1:])

  return pretrained

def tokens_to_chars(tokens, padding_value=PADDING_TOKEN):
  """Splits tokens into unicode characters.

  Args:
    tokens: A string ``tf.Tensor`` of shape :math:`[T]`.
    padding_value: The value to use for padding.

  Returns:
    The characters as a string ``tf.Tensor`` of shape :math:`[T, W]` and the
    length of each token as an int64 ``tf.Tensor``  of shape :math:`[T]`.
  """
  ragged = tf.strings.unicode_split(tokens, "UTF-8")
  chars = ragged.to_tensor(default_value=padding_value)
  lengths = ragged.row_lengths()
  return chars, lengths

def _get_field(config, key, prefix=None, default=None, required=False):
  if prefix:
    key = "%s%s" % (prefix, key)
  value = config.get(key, default)
  if value is None and required:
    raise ValueError("Missing field '%s' in the data configuration" % key)
  return value


@six.add_metaclass(abc.ABCMeta)
class TextInputter(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self, num_oov_buckets=1, dtype=tf.float32):
    super(TextInputter, self).__init__(dtype=dtype)
    self.num_oov_buckets = num_oov_buckets

  def initialize(self, data_config, asset_prefix=""):
    self.vocabulary_file = _get_field(
        data_config, "vocabulary", prefix=asset_prefix, required=True)
    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    tokenizer_config = _get_field(data_config, "tokenization", prefix=asset_prefix)
    self.tokenizer = tokenizers.make_tokenizer(tokenizer_config)

  def export_assets(self, asset_dir, asset_prefix=""):
    return self.tokenizer.export_assets(asset_dir, asset_prefix=asset_prefix)

  def vocabulary_lookup(self):
    """Returns a lookup table mapping string to index."""
    initializer = tf.lookup.TextFileInitializer(
        self.vocabulary_file,
        tf.string,
        TextFileIndex.WHOLE_LINE,
        tf.int64,
        TextFileIndex.LINE_NUMBER,
        vocab_size=self.vocabulary_size - self.num_oov_buckets)
    if self.num_oov_buckets > 0:
      return tf.lookup.StaticVocabularyTable(initializer, self.num_oov_buckets)
    else:
      return tf.lookup.StaticHashTable(initializer, 0)

  def vocabulary_lookup_reverse(self):
    """Returns a lookup table mapping index to string."""
    initializer = tf.lookup.TextFileInitializer(
        self.vocabulary_file,
        tf.int64,
        TextFileIndex.LINE_NUMBER,
        tf.string,
        TextFileIndex.WHOLE_LINE,
        vocab_size=self.vocabulary_size - self.num_oov_buckets)
    return tf.lookup.StaticHashTable(initializer, constants.UNKNOWN_TOKEN)

  def make_dataset(self, data_file, training=None):
    self.vocabulary = self.vocabulary_lookup()
    return tf.data.TextLineDataset(data_file)

  def make_features(self, element=None, features=None, training=None):
    """Tokenizes raw text."""
    if features is None:
      features = {}
    if "tokens" in features:
      return features
    tokens = self.tokenizer.tokenize(element)
    features["length"] = tf.shape(tokens)[0]
    features["tokens"] = tokens
    return features

  def input_signature(self):
    return {
        "tokens": tf.TensorSpec([None, None], tf.string),
        "length": tf.TensorSpec([None], tf.int32)
    }


class WordEmbedder(TextInputter):
  """Simple word embedder."""

  def __init__(self, embedding_size=None, dropout=0.0, dtype=tf.float32):
    """Initializes the parameters of the word embedder.

    Args:
      embedding_size: The size of the resulting embedding.
        If ``None``, an embedding file must be provided.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(WordEmbedder, self).__init__(dtype=dtype)
    self.embedding_size = embedding_size
    self.embedding_file = None
    self.dropout = dropout

  def initialize(self, data_config, asset_prefix=""):
    super(WordEmbedder, self).initialize(data_config, asset_prefix=asset_prefix)
    embedding = _get_field(data_config, "embedding", prefix=asset_prefix)
    if embedding is None and self.embedding_size is None:
      raise ValueError("embedding_size must be set")
    if embedding is not None:
      self.embedding_file = embedding["path"]
      self.trainable = embedding.get("trainable", True)
      self.embedding_file_with_header = embedding.get("with_header", True)
      self.case_insensitive_embeddings = embedding.get("case_insensitive", True)

  def make_features(self, element=None, features=None, training=None):
    """Converts words tokens to ids."""
    features = super(WordEmbedder, self).make_features(
        element=element, features=features, training=training)
    if "ids" in features:
      return features
    if self.vocabulary is None:
      self.vocabulary = self.vocabulary_lookup()
    ids = self.vocabulary.lookup(features["tokens"])
    if not self.is_target:
      features["ids"] = ids
    else:
      bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=ids.dtype)
      eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=ids.dtype)
      features["ids"] = tf.concat([bos, ids], axis=0)
      features["ids_out"] = tf.concat([ids, eos], axis=0)
      features["length"] += 1 # Increment length accordingly.
    return features

  def build(self, input_shape=None):
    if self.embedding_file:
      pretrained = load_pretrained_embeddings(
          self.embedding_file,
          self.vocabulary_file,
          num_oov_buckets=self.num_oov_buckets,
          with_header=self.embedding_file_with_header,
          case_insensitive_embeddings=self.case_insensitive_embeddings)
      self.embedding_size = pretrained.shape[-1]
      initializer = tf.constant_initializer(value=pretrained.astype(self.dtype))
    else:
      initializer = None
    self.embedding = self.add_weight(
        "embedding",
        [self.vocabulary_size, self.embedding_size],
        initializer=initializer,
        trainable=self.trainable)
    super(WordEmbedder, self).build(input_shape)

  def make_inputs(self, features, training=None):
    if not self.built:
      self.build()
    outputs = tf.nn.embedding_lookup(self.embedding, features["ids"])
    outputs = common.dropout(outputs, self.dropout, training=training)
    return outputs

  def visualize(self, log_dir):
    visualize_embeddings(
        log_dir,
        self.embedding,
        self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets)


@six.add_metaclass(abc.ABCMeta)
class CharEmbedder(TextInputter):
  """Base class for character-aware inputters."""

  def __init__(self, embedding_size, dropout=0.0, dtype=tf.float32):
    """Initializes the parameters of the character embedder.

    Args:
      embedding_size: The size of the character embedding.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(CharEmbedder, self).__init__(dtype=dtype)
    self.embedding_size = embedding_size
    self.embedding = None
    self.dropout = dropout

  def make_features(self, element=None, features=None, training=None):
    """Converts words to characters."""
    if features is None:
      features = {}
    if "char_ids" in features:
      return features
    if "chars" in features:
      chars = features["chars"]
    else:
      features = super(CharEmbedder, self).make_features(
          element=element, features=features, training=training)
      chars, _ = tokens_to_chars(features["tokens"])
    if self.vocabulary is None:
      self.vocabulary = self.vocabulary_lookup()
    features["char_ids"] = self.vocabulary.lookup(chars)
    return features

  def build(self, input_shape=None):
    self.embedding = self.add_weight(
        "char_embedding", [self.vocabulary_size, self.embedding_size])
    super(CharEmbedder, self).build(input_shape)

  @abc.abstractmethod
  def make_inputs(self, features, training=None):
    raise NotImplementedError()

  def visualize(self, log_dir):
    visualize_embeddings(
        log_dir,
        self.embedding,
        self.vocabulary_file,
        num_oov_buckets=self.num_oov_buckets)

  def _embed(self, inputs, training):
    if not self.built:
      self.build()
    mask = tf.math.not_equal(inputs, 0)
    outputs = tf.nn.embedding_lookup(self.embedding, inputs)
    outputs = common.dropout(outputs, self.dropout, training=training)
    return outputs, mask


class CharConvEmbedder(CharEmbedder):
  """An inputter that applies a convolution on characters embeddings."""

  def __init__(self,
               embedding_size,
               num_outputs,
               kernel_size=5,
               stride=3,
               dropout=0.0,
               dtype=tf.float32):
    """Initializes the parameters of the character convolution embedder.

    Args:
      embedding_size: The size of the character embedding.
      num_outputs: The dimension of the convolution output space.
      kernel_size: Length of the convolution window.
      stride: Length of the convolution stride.
      dropout: The probability to drop units in the embedding.
      dtype: The embedding type.
    """
    super(CharConvEmbedder, self).__init__(
        embedding_size,
        dropout=dropout,
        dtype=dtype)
    self.output_size = num_outputs
    self.conv = tf.keras.layers.Conv1D(
        num_outputs,
        kernel_size,
        strides=stride,
        padding="same")

  def make_inputs(self, features, training=None):
    inputs = features["char_ids"]
    flat_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
    outputs, _ = self._embed(flat_inputs, training)
    outputs = self.conv(outputs)
    outputs = tf.reduce_max(outputs, axis=1)
    outputs = tf.reshape(outputs, [-1, tf.shape(inputs)[1], self.output_size])
    return outputs


class CharRNNEmbedder(CharEmbedder):
  """An inputter that runs a single RNN layer over character embeddings."""

  def __init__(self,
               embedding_size,
               num_units,
               dropout=0.2,
               cell_class=None,
               dtype=tf.float32):
    """Initializes the parameters of the character RNN embedder.

    Args:
      embedding_size: The size of the character embedding.
      num_units: The number of units in the RNN layer.
      dropout: The probability to drop units in the embedding and the RNN
        outputs.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell. Defaults to a LSTM cell.
      dtype: The embedding type.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(CharRNNEmbedder, self).__init__(
        embedding_size,
        dropout=dropout,
        dtype=dtype)
    if cell_class is None:
      cell_class = tf.keras.layers.LSTMCell
    self.rnn = tf.keras.layers.RNN(cell_class(num_units))
    self.num_units = num_units

  def make_inputs(self, features, training=None):
    inputs = features["char_ids"]
    flat_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
    embeddings, mask = self._embed(flat_inputs, training)
    outputs = self.rnn(embeddings, mask=mask, training=training)
    outputs = tf.reshape(outputs, [-1, tf.shape(inputs)[1], self.num_units])
    return outputs
