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


def make_tokenizer(config=None):
  """Creates a tokenizer instance from the configuration.

  Args:
    config: Path to a configuration file or the configuration dictionary.

  Returns:
    A :class:`opennmt.tokenizers.Tokenizer` instance.

  Raises:
    ValueError: if :obj:`config` is invalid.
  """
  if config:
    if isinstance(config, six.string_types) and tf.io.gfile.exists(config):
      with tf.io.gfile.GFile(config, mode="rb") as config_file:
        config = yaml.load(config_file, Loader=yaml.UnsafeLoader)
    if isinstance(config, dict):
      tokenizer_type = config.get("type")
      tokenizer_params = config.get("params", {})
      if tokenizer_type is None:
        tokenizer = OpenNMTTokenizer(**config)
      else:
        tokenizer_class = getattr(sys.modules[__name__], tokenizer_type, None)
        if tokenizer_class is None:
          raise ValueError("Invalid tokenizer type: %s" % tokenizer_type)
        tokenizer = tokenizer_class(**tokenizer_params)
    else:
      raise ValueError("Invalid tokenization configuration: %s" % str(config))
  else:
    # If the tokenization was not configured, we assume that an external tokenization
    # was used and we don't include the tokenizer in the exported graph.
    tokenizer = SpaceTokenizer(in_graph=False)
  return tokenizer
