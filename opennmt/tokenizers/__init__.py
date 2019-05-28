"""Module defining tokenizers.

Tokenizers can work on string ``tf.Tensor`` as in-graph transformation.
"""

import sys
import six
import yaml

import tensorflow as tf

try:
  import pyonmttok
  from opennmt.tokenizers.opennmt_tokenizer import OpenNMTTokenizer
except ImportError:
  pass

from opennmt.tokenizers.tokenizer import Tokenizer
from opennmt.tokenizers.tokenizer import SpaceTokenizer
from opennmt.tokenizers.tokenizer import CharacterTokenizer


def make_tokenizer(config):
  """Creates a tokenizer instance from the configuration.

  Args:
    config: Path to a configuration file or the configuration dictionary.

  Returns:
    A :class:`opennmt.tokenizers.tokenizer.Tokenizer` instance.
  """
  if config:
    if isinstance(config, six.string_types) and tf.io.gfile.exists(config):
      with tf.io.gfile.GFile(config, mode="rb") as config_file:
        config = yaml.load(config_file, Loader=yaml.UnsafeLoader)
    tokenizer = OpenNMTTokenizer(**config)
  else:
    tokenizer = SpaceTokenizer()
  return tokenizer
