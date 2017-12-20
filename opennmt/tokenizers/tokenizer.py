"""Define base tokenizers."""

import os
import abc
import six
import yaml

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Tokenizer(object):
  """Base class for tokenizers."""

  def __init__(self, configuration_file_or_key=None):
    self._config = {}
    if configuration_file_or_key is not None and os.path.isfile(configuration_file_or_key):
      configuration_file = configuration_file_or_key
      with open(configuration_file) as conf_file:
        self._config = yaml.load(conf_file)
      self._configuration_file_key = None
    else:
      self._configuration_file_key = configuration_file_or_key

  def __call__(self, text):
    """Tokenizes text.

    Args:
      text: The text to tokenize as a ``tf.Tensor`` or Python string.

    Returns:
      A 1-D string ``tf.Tensor`` if :obj:`text` is a ``tf.Tensor`` or a list of
      Python unicode strings otherwise.
    """
    if tf.contrib.framework.is_tensor(text):
      return self._tokenize_tensor(text)
    else:
      text = tf.compat.as_text(text)
      return self._tokenize_string(text)

  def initialize(self, metadata):
    """Initializes the tokenizer (e.g. load BPE models).

    Any external assets should be registered in the standard assets collection:

    .. code-block:: python

        tf.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    if self._configuration_file_key is not None:
      self._config = metadata.get(self._configuration_file_key, {})

  def _tokenize_tensor(self, text):
    """Tokenizes a tensor.

    When not overriden, this default implementation uses a ``tf.py_func``
    operation to call the string-based tokenization.

    Args:
      text: A 1-D string ``tf.Tensor``.

    Returns:
      A 1-D string ``tf.Tensor``.
    """
    text = tf.py_func(
        lambda x: tf.compat.as_bytes("\0".join(self(x))), [text], tf.string)
    tokens = tf.string_split([text], delimiter="\0").values
    return tokens

  @abc.abstractmethod
  def _tokenize_string(self, text):
    """Tokenizes a Python unicode string.

    This method should be thread-safe.

    Args:
      text: A Python unicode string.

    Returns:
      A list of Python unicode strings.
    """
    raise NotImplementedError()


class SpaceTokenizer(Tokenizer):
  """A tokenizer that splits on spaces."""

  def _tokenize_tensor(self, text):
    return tf.string_split([text]).values

  def _tokenize_string(self, text):
    return text.split()


class CharacterTokenizer(Tokenizer):
  """A tokenizer that splits unicode characters."""

  def _tokenize_string(self, text):
    return list(text)
