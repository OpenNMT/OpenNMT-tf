"""Define word-based embedders."""

import abc
import collections
import os
import six

import numpy as np
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

from google.protobuf import text_format

from opennmt.tokenizers.tokenizer import SpaceTokenizer
from opennmt.inputters.inputter import Inputter
from opennmt.utils.cell import build_cell, last_encoding_from_state
from opennmt.utils.misc import count_lines
from opennmt.constants import PADDING_TOKEN
from opennmt.layers.common import embedding_lookup


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
  basename = os.path.basename(vocabulary_file)
  destination = os.path.join(log_dir, basename)
  if vocabulary_file != destination:
    tf.gfile.Copy(vocabulary_file, destination, overwrite=True)

    # Append <unk> tokens.
    with tf.gfile.Open(destination, mode="ab") as vocab:
      if num_oov_buckets == 1:
        vocab.write(b"<unk>\n")
      else:
        for i in range(num_oov_buckets):
          vocab.write(tf.compat.as_bytes("<unk%d>\n" % i))

  config = projector.ProjectorConfig()

  # If the projector file exists, load it.
  target = os.path.join(log_dir, "projector_config.pbtxt")
  if tf.gfile.Exists(target):
    with tf.gfile.Open(target, mode="rb") as target_file:
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
  with tf.gfile.Open(vocabulary_file, mode="rb") as vocabulary:
    count = 0
    for word in vocabulary:
      word = word.strip()
      if case_insensitive_embeddings:
        word = word.lower()
      word_to_id[word].append(count)
      count += 1

  # Fill pretrained embedding matrix.
  with tf.gfile.Open(embedding_file, mode="rb") as embedding:
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

def tokens_to_chars(tokens):
  """Splits a list of tokens into unicode characters.

  This is an in-graph transformation.

  Args:
    tokens: A sequence of tokens.

  Returns:
    The characters as a ``tf.Tensor`` of shape
    ``[sequence_length, max_word_length]`` and the length of each word.
  """

  def _split_chars(token, max_length, delimiter=" "):
    chars = list(token.decode("utf-8"))
    while len(chars) < max_length:
      chars.append(PADDING_TOKEN)
    return delimiter.join(chars).encode("utf-8")

  def _string_len(token):
    return len(token.decode("utf-8"))

  def _apply():
    # Get the length of each token.
    lengths = tf.map_fn(
        lambda x: tf.py_func(_string_len, [x], tf.int64),
        tokens,
        dtype=tf.int64,
        back_prop=False)

    max_length = tf.reduce_max(lengths)

    # Add a delimiter between each unicode character.
    spaced_chars = tf.map_fn(
        lambda x: tf.py_func(_split_chars, [x, max_length], [tf.string]),
        tokens,
        dtype=[tf.string],
        back_prop=False)

    # Split on this delimiter
    chars = tf.map_fn(
        lambda x: tf.string_split(x, delimiter=" ").values,
        spaced_chars,
        dtype=tf.string,
        back_prop=False)

    return chars, lengths

  def _none():
    chars = tf.constant([], dtype=tf.string)
    lengths = tf.constant([], dtype=tf.int64)
    return chars, lengths

  chars, lengths = tf.cond(tf.equal(tf.shape(tokens)[0], 0), true_fn=_none, false_fn=_apply)
  chars.set_shape([None, None])
  lengths.set_shape([None])
  return chars, lengths


@six.add_metaclass(abc.ABCMeta)
class TextInputter(Inputter):
  """An abstract inputter that processes text."""

  def __init__(self, tokenizer=SpaceTokenizer(), dtype=tf.float32):
    super(TextInputter, self).__init__(dtype=dtype)
    self.tokenizer = tokenizer

  def get_length(self, data):
    return data["length"]

  def make_dataset(self, data_file):
    return tf.data.TextLineDataset(data_file)

  def get_dataset_size(self, data_file):
    return count_lines(data_file)

  def initialize(self, metadata):
    self.tokenizer.initialize(metadata)

  def _process(self, data):
    """Tokenizes raw text."""
    data = super(TextInputter, self)._process(data)

    if "tokens" not in data:
      text = data["raw"]
      tokens = self.tokenizer.tokenize(text)
      length = tf.shape(tokens)[0]

      data = self.set_data_field(data, "tokens", tokens)
      data = self.set_data_field(data, "length", length)

    return data

  @abc.abstractmethod
  def _get_serving_input(self):
    raise NotImplementedError()

  @abc.abstractmethod
  def _transform_data(self, data, mode):
    raise NotImplementedError()

  @abc.abstractmethod
  def transform(self, inputs, mode):
    raise NotImplementedError()


class WordEmbedder(TextInputter):
  """Simple word embedder."""

  def __init__(self,
               vocabulary_file_key,
               embedding_size=None,
               embedding_file_key=None,
               embedding_file_with_header=True,
               case_insensitive_embeddings=True,
               trainable=True,
               dropout=0.0,
               tokenizer=SpaceTokenizer(),
               dtype=tf.float32):
    """Initializes the parameters of the word embedder.

    Args:
      vocabulary_file_key: The data configuration key of the vocabulary file
        containing one word per line.
      embedding_size: The size of the resulting embedding.
        If ``None``, an embedding file must be provided.
      embedding_file_key: The data configuration key of the embedding file.
      embedding_file_with_header: ``True`` if the embedding file starts with a
        header line like in GloVe embedding files.
      case_insensitive_embeddings: ``True`` if embeddings are trained on
        lowercase data.
      trainable: If ``False``, do not optimize embeddings.
      dropout: The probability to drop units in the embedding.
      tokenizer: An optional :class:`opennmt.tokenizers.tokenizer.Tokenizer` to
        tokenize the input text.
      dtype: The embedding type.

    Raises:
      ValueError: if neither :obj:`embedding_size` nor :obj:`embedding_file_key`
        are set.

    See Also:
      The :meth:`opennmt.inputters.text_inputter.load_pretrained_embeddings`
      function for details about the pretrained embedding format and behavior.
    """
    super(WordEmbedder, self).__init__(tokenizer=tokenizer, dtype=dtype)

    self.vocabulary_file_key = vocabulary_file_key
    self.embedding_size = embedding_size
    self.embedding_file_key = embedding_file_key
    self.embedding_file_with_header = embedding_file_with_header
    self.case_insensitive_embeddings = case_insensitive_embeddings
    self.trainable = trainable
    self.dropout = dropout
    self.num_oov_buckets = 1

    if embedding_size is None and embedding_file_key is None:
      raise ValueError("Must either provide embedding_size or embedding_file_key")

  def initialize(self, metadata):
    super(WordEmbedder, self).initialize(metadata)
    self.vocabulary_file = metadata[self.vocabulary_file_key]
    self.embedding_file = metadata[self.embedding_file_key] if self.embedding_file_key else None

    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    self.vocabulary = tf.contrib.lookup.index_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        num_oov_buckets=self.num_oov_buckets)

  def _get_serving_input(self):
    receiver_tensors = {
        "tokens": tf.placeholder(tf.string, shape=(None, None)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

    features = receiver_tensors.copy()
    features["ids"] = self.vocabulary.lookup(features["tokens"])

    return receiver_tensors, features

  def _process(self, data):
    """Converts words tokens to ids."""
    data = super(WordEmbedder, self)._process(data)

    if "ids" not in data:
      tokens = data["tokens"]
      ids = self.vocabulary.lookup(tokens)

      data = self.set_data_field(data, "ids", ids)

    return data

  def visualize(self, log_dir):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      embeddings = tf.get_variable("w_embs", dtype=self.dtype)
      visualize_embeddings(
          log_dir,
          embeddings,
          self.vocabulary_file,
          num_oov_buckets=self.num_oov_buckets)

  def _transform_data(self, data, mode):
    return self.transform(data["ids"], mode)

  def transform(self, inputs, mode):
    try:
      embeddings = tf.get_variable("w_embs", dtype=self.dtype, trainable=self.trainable)
    except ValueError:
      # Variable does not exist yet.
      if self.embedding_file:
        pretrained = load_pretrained_embeddings(
            self.embedding_file,
            self.vocabulary_file,
            num_oov_buckets=self.num_oov_buckets,
            with_header=self.embedding_file_with_header,
            case_insensitive_embeddings=self.case_insensitive_embeddings)
        self.embedding_size = pretrained.shape[-1]

        initializer = tf.constant_initializer(
            pretrained.astype(self.dtype.as_numpy_dtype()), dtype=self.dtype)
      else:
        initializer = None

      shape = [self.vocabulary_size, self.embedding_size]
      embeddings = tf.get_variable(
          "w_embs",
          shape=shape,
          dtype=self.dtype,
          initializer=initializer,
          trainable=self.trainable)

    outputs = embedding_lookup(embeddings, inputs)

    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)

    return outputs


@six.add_metaclass(abc.ABCMeta)
class CharEmbedder(TextInputter):
  """Base class for character-aware inputters."""

  def __init__(self,
               vocabulary_file_key,
               embedding_size,
               dropout=0.0,
               tokenizer=SpaceTokenizer(),
               dtype=tf.float32):
    """Initializes the parameters of the character embedder.

    Args:
      vocabulary_file_key: The meta configuration key of the vocabulary file
        containing one character per line.
      embedding_size: The size of the character embedding.
      dropout: The probability to drop units in the embedding.
      tokenizer: An optional :class:`opennmt.tokenizers.tokenizer.Tokenizer` to
        tokenize the input text.
      dtype: The embedding type.
    """
    super(CharEmbedder, self).__init__(tokenizer=tokenizer, dtype=dtype)

    self.vocabulary_file_key = vocabulary_file_key
    self.embedding_size = embedding_size
    self.dropout = dropout
    self.num_oov_buckets = 1

  def initialize(self, metadata):
    super(CharEmbedder, self).initialize(metadata)
    self.vocabulary_file = metadata[self.vocabulary_file_key]
    self.vocabulary_size = count_lines(self.vocabulary_file) + self.num_oov_buckets
    self.vocabulary = tf.contrib.lookup.index_table_from_file(
        self.vocabulary_file,
        vocab_size=self.vocabulary_size - self.num_oov_buckets,
        num_oov_buckets=self.num_oov_buckets)

  def _get_serving_input(self):
    receiver_tensors = {
        "chars": tf.placeholder(tf.string, shape=(None, None, None)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

    features = receiver_tensors.copy()
    features["char_ids"] = self.vocabulary.lookup(features["chars"])

    return receiver_tensors, features

  def _process(self, data):
    """Converts words to characters."""
    data = super(CharEmbedder, self)._process(data)

    if "char_ids" not in data:
      tokens = data["tokens"]
      chars, _ = tokens_to_chars(tokens)
      ids = self.vocabulary.lookup(chars)

      data = self.set_data_field(data, "char_ids", ids)

    return data

  def visualize(self, log_dir):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      embeddings = tf.get_variable("w_char_embs", dtype=self.dtype)
      visualize_embeddings(
          log_dir,
          embeddings,
          self.vocabulary_file,
          num_oov_buckets=self.num_oov_buckets)

  def _transform_data(self, data, mode):
    return self.transform(data["char_ids"], mode)

  def _embed(self, inputs, mode):
    embeddings = tf.get_variable(
        "w_char_embs", shape=[self.vocabulary_size, self.embedding_size], dtype=self.dtype)

    outputs = embedding_lookup(embeddings, inputs)
    outputs = tf.layers.dropout(
        outputs,
        rate=self.dropout,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    return outputs

  @abc.abstractmethod
  def transform(self, inputs, mode):
    raise NotImplementedError()


class CharConvEmbedder(CharEmbedder):
  """An inputter that applies a convolution on characters embeddings."""

  def __init__(self,
               vocabulary_file_key,
               embedding_size,
               num_outputs,
               kernel_size=5,
               stride=3,
               dropout=0.0,
               tokenizer=SpaceTokenizer(),
               dtype=tf.float32):
    """Initializes the parameters of the character convolution embedder.

    Args:
      vocabulary_file_key: The meta configuration key of the vocabulary file
        containing one character per line.
      embedding_size: The size of the character embedding.
      num_outputs: The dimension of the convolution output space.
      kernel_size: Length of the convolution window.
      stride: Length of the convolution stride.
      dropout: The probability to drop units in the embedding.
      tokenizer: An optional :class:`opennmt.tokenizers.tokenizer.Tokenizer` to
        tokenize the input text.
      dtype: The embedding type.
    """
    super(CharConvEmbedder, self).__init__(
        vocabulary_file_key,
        embedding_size,
        dropout=dropout,
        tokenizer=tokenizer,
        dtype=dtype)
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.stride = stride
    self.num_oov_buckets = 1

  def transform(self, inputs, mode):
    outputs = self._embed(inputs, mode)

    # Merge batch and sequence timesteps dimensions.
    outputs = tf.reshape(outputs, [-1, tf.shape(inputs)[-1], self.embedding_size])

    # Pad on both sides.
    outputs = tf.pad(outputs, [[0, 0], [self.kernel_size - 1, self.kernel_size - 1], [0, 0]])
    outputs.set_shape((None, None, self.embedding_size))

    outputs = tf.layers.conv1d(
        outputs,
        self.num_outputs,
        self.kernel_size,
        strides=self.stride)

    # Max pooling over depth.
    outputs = tf.reduce_max(outputs, axis=1)

    # Split batch and sequence timesteps dimensions.
    outputs = tf.reshape(outputs, [-1, tf.shape(inputs)[1], self.num_outputs])

    return outputs


class CharRNNEmbedder(CharEmbedder):
  """An inputter that runs a single RNN layer over character embeddings."""

  def __init__(self,
               vocabulary_file_key,
               embedding_size,
               num_units,
               dropout=0.2,
               encoding="average",
               cell_class=tf.contrib.rnn.LSTMCell,
               tokenizer=SpaceTokenizer(),
               dtype=tf.float32):
    """Initializes the parameters of the character RNN embedder.

    Args:
      vocabulary_file_key: The meta configuration key of the vocabulary file
        containing one character per line.
      embedding_size: The size of the character embedding.
      num_units: The number of units in the RNN layer.
      dropout: The probability to drop units in the embedding and the RNN
        outputs.
      encoding: "average" or "last" (case insensitive), the encoding vector to
        extract from the RNN outputs.
      cell_class: The inner cell class or a callable taking :obj:`num_units` as
        argument and returning a cell.
      tokenizer: An optional :class:`opennmt.tokenizers.tokenizer.Tokenizer` to
        tokenize the input text.
      dtype: The embedding type.

    Raises:
      ValueError: if :obj:`encoding` is invalid.
    """
    super(CharRNNEmbedder, self).__init__(
        vocabulary_file_key,
        embedding_size,
        dropout=dropout,
        tokenizer=tokenizer,
        dtype=dtype)
    self.num_units = num_units
    self.cell_class = cell_class
    self.encoding = encoding.lower()
    if self.encoding not in ("average", "last"):
      raise ValueError("Invalid encoding vector: {}".format(self.encoding))

  def transform(self, inputs, mode):
    flat_inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[-1]])
    embeddings = self._embed(flat_inputs, mode)
    sequence_length = tf.count_nonzero(flat_inputs, axis=1)

    cell = build_cell(
        1,
        self.num_units,
        mode,
        dropout=self.dropout,
        cell_class=self.cell_class)
    rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
        cell,
        embeddings,
        sequence_length=sequence_length,
        dtype=embeddings.dtype)

    if self.encoding == "average":
      encoding = tf.reduce_mean(rnn_outputs, axis=1)
    elif self.encoding == "last":
      encoding = last_encoding_from_state(rnn_state)

    outputs = tf.reshape(encoding, [-1, tf.shape(inputs)[1], self.num_units])
    return outputs
