"""Vocabulary utilities for Python scripts."""

import six

import tensorflow as tf

from opennmt.utils import compat


class Vocab(object):
  """Vocabulary class."""

  def __init__(self, special_tokens=None, from_file=None, from_format="default"):
    """Initializes a vocabulary.

    Args:
      special_tokens: A list of special tokens (e.g. start of sentence).
      from_file: Optionally initialize from an existing saved vocabulary.
      from_format: Define the format of the :obj:`from_file` saved vocabulary.
        Can be: default, sentencepiece. "default" is simply one token per line.

    Raises:
      ValueError: if :obj:`file_format` is invalid.
    """
    self._token_to_id = {}
    self._id_to_token = []
    self._frequency = []

    if special_tokens is not None:
      for index, token in enumerate(special_tokens):
        self._token_to_id[token] = index
        self._id_to_token.insert(index, token)

        # Set a very high frequency to avoid special tokens to be pruned. Note that Python sort
        # functions are stable which means that special tokens in pruned vocabularies will have
        # the same index.
        self._frequency.insert(index, float("inf"))

    if from_file is not None:
      self.load(from_file, file_format=from_format)

  @property
  def size(self):
    """Returns the number of entries of the vocabulary."""
    return len(self._id_to_token)

  @property
  def words(self):
    """Returns the list of words."""
    return self._id_to_token

  def __len__(self):
    """Returns the number of entries of the vocabulary."""
    return self.size

  def __contains__(self, token):
    """Returns ``True`` if the vocabulary contains :obj:`token`."""
    return self.lookup(token) is not None

  def add_from_text(self, filename, tokenizer=None):
    """Fills the vocabulary from a text file.

    Args:
      filename: The file to load from.
      tokenizer: A callable to tokenize a line of text.
    """
    with compat.gfile_open(filename, mode="rb") as text:
      for line in text:
        line = tf.compat.as_text(line.strip())
        if tokenizer:
          tokens = tokenizer.tokenize(line)
        else:
          tokens = line.split()
        for token in tokens:
          self.add(token)

  def serialize(self, path):
    """Writes the vocabulary on disk.

    Args:
      path: The path where the vocabulary will be saved.
    """
    with compat.gfile_open(path, mode="wb") as vocab:
      for token in self._id_to_token:
        vocab.write(tf.compat.as_bytes(token))
        vocab.write(b"\n")

  def load(self, path, file_format="default"):
    """Loads a serialized vocabulary.

    Args:
      path: The path to the vocabulary to load.
      file_format: Define the format of the vocabulary file. Can be: default,
        sentencepiece. "default" is simply one token per line.

    Raises:
      ValueError: if :obj:`file_format` is invalid.
    """
    with compat.gfile_open(path, mode="rb") as vocab:
      for line in vocab:
        if file_format == "default":
          self.add(line[:-1])
        elif file_format == "sentencepiece":
          token, _ = line.rstrip().split(b"\t")
          if token in (b"<unk>", b"<s>", b"</s>"):  # Ignore SentencePiece special tokens.
            continue
          self.add(token)
        else:
          raise ValueError("Invalid vocabulary format: %s" % file_format)

  def add(self, token):
    """Adds a token or increases its frequency.

    Args:
      token: The string to add.
    """
    if isinstance(token, six.binary_type):
      token = tf.compat.as_text(token)
    if token not in self._token_to_id:
      index = self.size
      self._token_to_id[token] = index
      self._id_to_token.append(token)
      self._frequency.append(1)
    else:
      self._frequency[self._token_to_id[token]] += 1

  def lookup(self, identifier, default=None):
    """Lookups in the vocabulary.

    Args:
      identifier: A string or an index to lookup.
      default: The value to return if :obj:`identifier` is not found.

    Returns:
      The value associated with :obj:`identifier` or :obj:`default`.
    """
    value = None

    if isinstance(identifier, six.string_types):
      if isinstance(identifier, six.binary_type):
        identifier = tf.compat.as_text(identifier)
      if identifier in self._token_to_id:
        value = self._token_to_id[identifier]
    elif identifier < self.size:
      value = self._id_to_token[identifier]

    if value is None:
      return default
    else:
      return value

  def prune(self, max_size=0, min_frequency=1):
    """Creates a pruned version of the vocabulary.

    Args:
      max_size: The maximum vocabulary size.
      min_frequency: The minimum frequency of each entry.

    Returns:
      A new vocabulary.
    """
    sorted_ids = sorted(range(self.size), key=lambda k: self._frequency[k], reverse=True)
    new_size = len(sorted_ids)

    # Discard words that do not meet frequency requirements.
    for i in range(new_size - 1, 0, -1):
      index = sorted_ids[i]
      if self._frequency[index] < min_frequency:
        new_size -= 1
      else:
        break

    # Limit absolute size.
    if max_size > 0:
      new_size = min(new_size, max_size)

    new_vocab = Vocab()

    for i in range(new_size):
      index = sorted_ids[i]
      token = self._id_to_token[index]
      frequency = self._frequency[index]

      new_vocab._token_to_id[token] = i  # pylint: disable=protected-access
      new_vocab._id_to_token.append(token)  # pylint: disable=protected-access
      new_vocab._frequency.append(frequency)  # pylint: disable=protected-access

    return new_vocab

  def pad_to_multiple(self, multiple, num_oov_buckets=1):
    """Pads the vocabulary size to a multiple value.

    More specically, this method ensures that:

        ``(vocab_size + num_oov_buckets) % multiple == 0``

    Args:
      multiple: The multiple value.
      num_oov_buckets: The number of OOV buckets added during the training.
        Usually just 1 for the `<unk>` token.
    """
    i = 0
    while (self.size + num_oov_buckets) % multiple != 0:
      self.add("averyunlikelytoken%d" % i)
      i += 1
