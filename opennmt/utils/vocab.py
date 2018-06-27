"""Vocabulary utilities for Python scripts."""

import six

import tensorflow as tf


class Vocab(object):
  """Vocabulary class."""

  def __init__(self, special_tokens=None, from_file=None):
    """Initializes a vocabulary.

    Args:
      special_tokens: A list of special tokens (e.g. start of sentence).
      from_file: Optionally initialize from an existing saved vocabulary.
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
      self.load(from_file)

  @property
  def size(self):
    """Returns the number of entries of the vocabulary."""
    return len(self._id_to_token)

  @property
  def words(self):
    """Returns the list of words."""
    return self._id_to_token

  def add_from_text(self, filename, tokenizer=None):
    """Fills the vocabulary from a text file.

    Args:
      filename: The file to load from.
      tokenizer: A callable to tokenize a line of text.
    """
    with open(filename, "rb") as text:
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
    with open(path, "wb") as vocab:
      for token in self._id_to_token:
        vocab.write(tf.compat.as_bytes(token))
        vocab.write(b"\n")

  def load(self, path):
    """Loads a serialized vocabulary.

    Args:
      path: The path to the vocabulary to load.
    """
    with open(path, "rb") as vocab:
      for token in vocab:
        token = token.strip()
        self.add(tf.compat.as_text(token))

  def add(self, token):
    """Adds a token or increases its frequency.

    Args:
      token: The string to add.
    """
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
