"""Module defining tokenizers.

Tokenizers can work on string ``tf.Tensor`` as in-graph transformation.
"""

import sys

try:
  import pyonmttok
  from opennmt.tokenizers.opennmt_tokenizer import OpenNMTTokenizer
except ImportError:
  pass

from opennmt.tokenizers.tokenizer import SpaceTokenizer, CharacterTokenizer

def add_command_line_arguments(parser):
  """Adds command line arguments to select the tokenizer."""
  from opennmt.utils.misc import classes_in_module
  choices = list(classes_in_module(sys.modules[__name__]))

  parser.add_argument(
      "--tokenizer", default="SpaceTokenizer", choices=choices,
      help="Tokenizer class name.")
  parser.add_argument(
      "--tokenizer_config", default=None,
      help="Tokenization configuration file.")

def build_tokenizer(args):
  """Returns a new tokenizer based on command line arguments."""
  module = sys.modules[__name__]
  return getattr(module, args.tokenizer)(configuration_file_or_key=args.tokenizer_config)
