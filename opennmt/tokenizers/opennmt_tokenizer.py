"""Define the OpenNMT tokenizer."""

import copy

import tensorflow as tf

import pyonmttok

from opennmt.tokenizers.tokenizer import Tokenizer


def create_tokenizer(config):
  """Creates a new OpenNMT tokenizer.

  Args:
    config: A dictionary of tokenization options.

  Returns:
    A ``pyonmttok.Tokenizer``.
  """
  kwargs = copy.deepcopy(config)
  mode = "conservative"
  if "mode" in kwargs:
    mode = kwargs["mode"]
    del kwargs["mode"]
  return pyonmttok.Tokenizer(mode, **kwargs)


class OpenNMTTokenizer(Tokenizer):
  """Uses the OpenNMT tokenizer."""

  def __init__(self, configuration_file_or_key=None):
    super(OpenNMTTokenizer, self).__init__(configuration_file_or_key=configuration_file_or_key)
    self._tokenizer = None

  def _tokenize_string(self, text):
    if self._tokenizer is None:
      self._tokenizer = create_tokenizer(self._config)
    text = tf.compat.as_bytes(text)
    tokens, _ = self._tokenizer.tokenize(text)
    return [tf.compat.as_text(token) for token in tokens]

  def _detokenize_string(self, tokens):
    if self._tokenizer is None:
      self._tokenizer = create_tokenizer(self._config)
    tokens = [tf.compat.as_bytes(token) for token in tokens]
    return tf.compat.as_text(self._tokenizer.detokenize(tokens))
