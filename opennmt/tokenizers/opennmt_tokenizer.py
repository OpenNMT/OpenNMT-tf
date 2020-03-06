"""Define the OpenNMT tokenizer."""

import os
import yaml

import tensorflow as tf

import pyonmttok

from opennmt.tokenizers import tokenizer


@tokenizer.register_tokenizer
class OpenNMTTokenizer(tokenizer.Tokenizer):
  """Uses the OpenNMT tokenizer."""

  def __init__(self, **kwargs):
    kwargs.setdefault("mode", "conservative")
    self._config = kwargs
    self._tokenizer = pyonmttok.Tokenizer(**kwargs)

  def export_assets(self, asset_dir, asset_prefix=""):
    assets = {}

    # Extract asset files from the configuration.
    config = self._config.copy()
    for key, value in config.items():
      if isinstance(value, str) and tf.io.gfile.exists(value):
        basename = os.path.basename(value)
        config[key] = basename  # Only save the basename.
        assets[basename] = value

    # Save the tokenizer configuration.
    config_name = "%stokenizer_config.yml" % asset_prefix
    config_path = os.path.join(asset_dir, config_name)
    assets[config_name] = config_path
    with tf.io.gfile.GFile(config_path, "w") as config_file:
      yaml.dump(config, stream=config_file, default_flow_style=False)

    return assets

  def _tokenize_string(self, text):
    tokens, _ = self._tokenizer.tokenize(text)
    return tokens

  def _detokenize_string(self, tokens):
    return self._tokenizer.detokenize(tokens)
