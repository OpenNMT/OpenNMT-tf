"""Define the OpenNMT tokenizer."""

import six

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
  def _set(kwargs, key):
    if key in config:
      value = config[key]
      if isinstance(value, six.string_types):
        value = tf.compat.as_bytes(value)
      kwargs[key] = value

  kwargs = {}
  _set(kwargs, "bpe_model_path")
  _set(kwargs, "sp_model_path")
  _set(kwargs, "joiner")
  _set(kwargs, "joiner_annotate")
  _set(kwargs, "joiner_new")
  _set(kwargs, "spacer_annotate")
  _set(kwargs, "case_feature")
  _set(kwargs, "no_substitution")
  _set(kwargs, "segment_case")
  _set(kwargs, "segment_numbers")
  _set(kwargs, "segment_alphabet_change")
  _set(kwargs, "segment_alphabet")

  return pyonmttok.Tokenizer(config.get("mode", "conservative"), **kwargs)


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
    return tokens

  def _detokenize_string(self, tokens):
    if self._tokenizer is None:
      self._tokenizer = create_tokenizer(self._config)
    tokens = [tf.compat.as_bytes(token) for token in tokens]
    return self._tokenizer.detokenize(tokens)
