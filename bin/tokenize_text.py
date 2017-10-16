"""Standalone script to tokenize a corpus."""

from __future__ import print_function

import argparse
import sys

from opennmt import tokenizers


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--tokenizer", default="SpaceTokenizer",
      help="Tokenizer class name.")
  parser.add_argument(
      "--delimiter", default=" ",
      help="Token delimiter for text serialization.")
  args = parser.parse_args()

  tokenizer = getattr(tokenizers, args.tokenizer)()

  for line in sys.stdin:
    line = line.strip().decode("utf-8")
    tokens = tokenizer(line)
    merged_tokens = args.delimiter.join(tokens)
    print(merged_tokens)

if __name__ == "__main__":
  main()
