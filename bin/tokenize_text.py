"""Standalone script to tokenize a corpus."""

from __future__ import print_function

import argparse
import sys

from opennmt import tokenizers


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
    tokens = tokenizer(line)
    merged_tokens = args.delimiter.join(tokens)
    print(merged_tokens)

if __name__ == "__main__":
  main()
