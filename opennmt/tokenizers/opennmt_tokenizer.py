"""Define the OpenNMT tokenizer."""

import os
import copy
import six
import yaml

import tensorflow as tf

import pyonmttok

from opennmt.tokenizers.tokenizer import Tokenizer


def _make_config_asset_file(config, asset_path):
  asset_config = copy.deepcopy(config)
  for key, value in six.iteritems(asset_config):
    # Only keep the basename for files (that should also be registered as assets).
    if isinstance(value, six.string_types) and tf.io.gfile.exists(value):
      asset_config[key] = os.path.basename(value)
  with tf.io.gfile.GFile(asset_path, "w") as asset_file:
    yaml.dump(asset_config, stream=asset_file, default_flow_style=False)


class OpenNMTTokenizer(Tokenizer):
  """Uses the OpenNMT tokenizer."""

  def __init__(self, **kwargs):
    self._config = copy.deepcopy(kwargs)
    mode = "conservative"
    if "mode" in kwargs:
      mode = kwargs["mode"]
      del kwargs["mode"]
    self._tokenizer = pyonmttok.Tokenizer(mode, **kwargs)

  def export_assets(self, asset_dir, asset_prefix=""):
    asset_name = "%stokenizer_config.yml" % asset_prefix
    asset_path = os.path.join(asset_dir, asset_name)
    _make_config_asset_file(self._config, asset_path)
    assets = {}
    assets[asset_name] = asset_path
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
