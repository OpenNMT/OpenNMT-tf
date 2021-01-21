"""Vocabulary utilities for Python scripts."""

import tensorflow as tf
import numpy as np

from opennmt import constants


class Vocab(object):
    """Vocabulary class.

    Example:

      >>> vocab = opennmt.data.Vocab.from_file("wmtende.vocab")
      >>> len(vocab)
      32000
      >>> "be" in vocab
      True
      >>> vocab.lookup("be")
      377
      >>> vocab.lookup(377)
      'be'
    """

    def __init__(self, special_tokens=None):
        """Initializes a vocabulary.

        Args:
          special_tokens: A list of special tokens (e.g. start of sentence).
        """
        self._token_to_id = {}
        self._id_to_token = []
        self._frequency = []

        if special_tokens is not None:
            for index, token in enumerate(special_tokens):
                self._token_to_id[token] = index
                self._id_to_token.insert(index, token)

                # Set a very high frequency to avoid special tokens to be pruned.
                # Note that Python sort functions are stable which means that
                # special tokens in pruned vocabularies will have the same index.
                self._frequency.insert(index, float("inf"))

    @classmethod
    def from_file(cls, path, file_format="default"):
        """Creates from a vocabulary file.

        Args:
          path: The path to the vocabulary file.
          file_format: Define the format of the vocabulary file. Can be: default,
            sentencepiece. "default" is simply one token per line.

        Raises:
          ValueError: if :obj:`file_format` is invalid.
        """
        vocab = cls()
        vocab.load(path, file_format=file_format)
        return vocab

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
        with tf.io.gfile.GFile(filename) as text:
            for line in text:
                line = line.rstrip("\r\n")
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
        with tf.io.gfile.GFile(path, mode="w") as vocab:
            for token in self._id_to_token:
                vocab.write(token)
                vocab.write("\n")

    def load(self, path, file_format="default"):
        """Loads a serialized vocabulary.

        Args:
          path: The path to the vocabulary to load.
          file_format: Define the format of the vocabulary file. Can be: default,
            sentencepiece. "default" is simply one token per line.

        Raises:
          ValueError: if :obj:`file_format` is invalid.
        """
        with tf.io.gfile.GFile(path) as vocab:
            for line in vocab:
                token = line.rstrip("\r\n")
                if file_format == "default":
                    self.add(token)
                elif file_format == "sentencepiece":
                    token, _ = token.split("\t")
                    if token in (
                        "<unk>",
                        "<s>",
                        "</s>",
                    ):  # Ignore SentencePiece special tokens.
                        continue
                    self.add(token)
                else:
                    raise ValueError("Invalid vocabulary format: %s" % file_format)

    def add(self, token):
        """Adds a token or increases its frequency.

        Args:
          token: The string to add.
        """
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

        if isinstance(identifier, (bytes, str)):
            identifier = tf.compat.as_text(identifier)
            value = self._token_to_id.get(identifier)
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
        sorted_ids = sorted(
            range(self.size), key=lambda k: self._frequency[k], reverse=True
        )
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

            new_vocab._token_to_id[token] = i
            new_vocab._id_to_token.append(token)
            new_vocab._frequency.append(frequency)

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


def create_lookup_tables(
    vocabulary_path, num_oov_buckets=1, as_asset=True, unk_token=None
):
    """Creates TensorFlow lookup tables from a vocabulary file.

    Args:
      vocabulary_path: Path to the vocabulary file.
      num_oov_buckets: Number of out-of-vocabulary buckets.
      as_asset: If ``True``, the vocabulary file will be added as a graph asset.
        Otherwise, the content of the vocabulary will be embedded in the graph.
      unk_token: The out-of-vocabulary token. Defaults to ``<unk>``.

    Returns:
      A tuple containing,

      - The final vocabulary size.
      - The ``tf.lookup`` table mapping tokens to ids.
      - The ``tf.lookup`` table mapping ids to tokens.
    """
    if unk_token is None:
        unk_token = constants.UNKNOWN_TOKEN
    vocabulary = Vocab.from_file(vocabulary_path)
    vocabulary_size = len(vocabulary)
    if as_asset:
        tokens_to_ids_initializer = tf.lookup.TextFileInitializer(
            vocabulary_path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            vocab_size=vocabulary_size,
        )
        ids_to_tokens_initializer = tf.lookup.TextFileInitializer(
            vocabulary_path,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            vocab_size=vocabulary_size,
        )
    else:
        tokens = tf.constant(vocabulary.words, dtype=tf.string)
        ids = tf.constant(list(range(vocabulary_size)), dtype=tf.int64)
        tokens_to_ids_initializer = tf.lookup.KeyValueTensorInitializer(tokens, ids)
        ids_to_tokens_initializer = tf.lookup.KeyValueTensorInitializer(ids, tokens)
    if num_oov_buckets > 0:
        tokens_to_ids = tf.lookup.StaticVocabularyTable(
            tokens_to_ids_initializer, num_oov_buckets
        )
    else:
        tokens_to_ids = tf.lookup.StaticHashTable(tokens_to_ids_initializer, 0)
    ids_to_tokens = tf.lookup.StaticHashTable(ids_to_tokens_initializer, unk_token)
    return vocabulary_size + num_oov_buckets, tokens_to_ids, ids_to_tokens


def get_mapping(current_vocab_path, new_vocab_path, mode="replace"):
    """Maps vocabulary new indices to old ones. -1 means that the entry is new."""
    mode = mode.lower()
    if mode not in ("merge", "replace"):
        raise ValueError("invalid vocab update mode: %s" % mode)
    current_vocab = Vocab.from_file(current_vocab_path)
    new_vocab = Vocab.from_file(new_vocab_path)
    mapping = []
    if mode == "merge":
        final_vocab = Vocab.from_file(current_vocab_path)
        mapping = list(range(current_vocab.size))
        for new_word in new_vocab.words:
            if current_vocab.lookup(new_word) is None:
                mapping.append(-1)
                final_vocab.add(new_word)
    elif mode == "replace":
        final_vocab = new_vocab
        for new_word in new_vocab.words:
            idx = current_vocab.lookup(new_word)
            if idx is not None:
                mapping.append(idx)
            else:
                mapping.append(-1)
    mapping.append(current_vocab.size)  # <unk> token is always the last entry.
    return mapping, final_vocab


def update_variable(ref_variable, new_variable, mapping, vocab_axis=0):
    """Update a vocabulary variable, possibly copying previous entries based on
    mapping.
    """
    ref = ref_variable.numpy()
    new = np.zeros(
        new_variable.shape.as_list(), dtype=new_variable.dtype.as_numpy_dtype
    )
    perm = None
    if vocab_axis != 0:
        # Make the dimension to index the first.
        perm = list(range(len(ref.shape)))
        perm[0], perm[vocab_axis] = perm[vocab_axis], perm[0]
        ref = np.transpose(ref, axes=perm)
        new = np.transpose(new, axes=perm)
    for i, j in enumerate(mapping):
        if j >= 0:
            new[i] = ref[j]
    if perm is not None:
        new = np.transpose(new, axes=perm)
    new_variable.assign(new)
    return new_variable


def update_variable_and_slots(
    ref_variable, new_variable, ref_optimizer, new_optimizer, mapping, vocab_axis=0
):
    """Update a vocabulary variable and its associated optimizer slots (if any)."""
    variables = []
    variables.append(
        update_variable(ref_variable, new_variable, mapping, vocab_axis=vocab_axis)
    )
    ref_slot_names = ref_optimizer.get_slot_names()
    new_slot_names = new_optimizer.get_slot_names()
    for slot_name in ref_slot_names:
        if slot_name not in new_slot_names:
            continue
        ref_slot = ref_optimizer.get_slot(ref_variable, slot_name)
        new_slot = new_optimizer.get_slot(new_variable, slot_name)
        variables.append(
            update_variable(ref_slot, new_slot, mapping, vocab_axis=vocab_axis)
        )
    return variables
