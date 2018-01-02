"""Standalone script to tokenize a corpus."""

from __future__ import print_function

import argparse

from opennmt import tokenizers


def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      "--delimiter", default=" ",
      help="Token delimiter for text serialization.")
  tokenizers.add_command_line_arguments(parser)
  args = parser.parse_args()

  tokenizer = tokenizers.build_tokenizer(args)
  tokenizer.tokenize_stream(delimiter=args.delimiter)

if __name__ == "__main__":
  main()
