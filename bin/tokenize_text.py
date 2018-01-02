"""Standalone script to tokenize a corpus."""

from __future__ import print_function

import argparse
import sys

import tensorflow as tf

from opennmt import tokenizers
from opennmt.utils.misc import print_bytes

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--delimiter", default=" ",
      help="Token delimiter for text serialization.")
  tokenizers.add_command_line_arguments(parser)
  args = parser.parse_args()

  tokenizer = tokenizers.build_tokenizer(args)

  for line in sys.stdin:
    line = line.strip()
    tokens = tokenizer.tokenize(line)
    merged_tokens = args.delimiter.join(tokens)
    print_bytes(tf.compat.as_bytes(merged_tokens))

if __name__ == "__main__":
  main()
