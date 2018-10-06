"""Define the OpenNMT tokenizer."""

import os
import copy
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
  kwargs = copy.deepcopy(config)
  mode = "conservative"
  if "mode" in kwargs:
    mode = kwargs["mode"]
    del kwargs["mode"]
  return pyonmttok.Tokenizer(mode, **kwargs)


class OpenNMTTokenizer(Tokenizer):
  """Uses the OpenNMT tokenizer."""

  def __init__(self, *arg, **kwargs):
    super(OpenNMTTokenizer, self).__init__(*arg, **kwargs)
    self._tokenizer = create_tokenizer(self._config)

  def initialize(self, metadata, asset_dir=None, asset_prefix=""):
    assets = super(OpenNMTTokenizer, self).initialize(
        metadata, asset_dir=asset_dir, asset_prefix=asset_prefix)
    self._tokenizer = create_tokenizer(self._config)
    if asset_dir is not None:
      for key, value in six.iteritems(self._config):
        if key.endswith("path"):
          assets[os.path.basename(value)] = value
    return assets

  def _tokenize_string(self, text):
    text = tf.compat.as_bytes(text)
    tokens, _ = self._tokenizer.tokenize(text)
    return [tf.compat.as_text(token) for token in tokens]

  def _detokenize_string(self, tokens):
    tokens = [tf.compat.as_bytes(token) for token in tokens]
    return tf.compat.as_text(self._tokenizer.detokenize(tokens))
