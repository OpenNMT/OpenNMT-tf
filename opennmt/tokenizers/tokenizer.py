"""Define base tokenizers."""

import abc
import six

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Tokenizer(object):
  """Base class for tokenizers."""

  def __call__(self, text):
    """Tokenizes text.

    Args:
      text: The text to tokenize as a `tf.Tensor` or Python string.

    Returns:
      A 1-D string `tf.Tensor` if `text` is a `tf.Tensor` or a list of Python
      strings otherwise.
    """
    if tf.contrib.framework.is_tensor(text):
      return self._tokenize_tensor(text)
    else:
      return self._tokenize_string(text)

  def initialize(self, metadata):
    """Initializes the tokenizer (e.g. load BPE models).

    Any external assets should be registered in the standard assets collection:

    ```
    tf.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
    ```

    Args:
      metadata: A dictionary containing additional metadata set
        by the user.
    """
    pass

  def _tokenize_tensor(self, text):
    """Tokenizes a tensor.

    When not overriden, this default implementation calls  `_tokenize_string`
    by using a `tf.py_func` operation.

    This method should be thread-safe.

    Args:
      text: A 1-D string `tf.Tensor`.

    Returns:
      A 1-D string `tf.Tensor`.
    """
    def _unicode_wrapper(text):
      unicode_text = text.decode("utf-8")
      unicode_tokens = self._tokenize_string(unicode_text)
      return " ".join(unicode_tokens).encode("utf-8")

    text = tf.py_func(_unicode_wrapper, [text], tf.string)
    tokens = tf.string_split([text]).values
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
