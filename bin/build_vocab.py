"""Standalone script to generate word vocabularies from monolingual corpus."""

import argparse
import os
import sys

from opennmt import constants
from opennmt import tokenizers


def main():
  parser = argparse.ArgumentParser(description="Generate vocabulary file")
  parser.add_argument(
      "data",
      help="Source text file.")
  parser.add_argument(
      "--save_vocab", required=True,
      help="Output vocabulary file.")
  parser.add_argument(
      "--tokenizer", default="SpaceTokenizer",
      help="Tokenizer class name.")
  parser.add_argument(
      "--min_frequency", type=int, default=1,
      help="Minimum word frequency.")
  parser.add_argument(
      "--size", type=int, default=0,
      help="Maximum vocabulary size. If = 0, do not limit vocabulary.")
  parser.add_argument(
      "--with_sequence_tokens", type=bool, default=True,
      help="If True, also inject special sequence tokens (start, end).")
  args = parser.parse_args()

  token_to_id = {}
  id_to_token = []
  frequency = []

  def add_token(token):
    if token not in token_to_id:
      index = len(id_to_token)
      token_to_id[token] = index
      id_to_token.append(token)
      frequency.append(1)
    else:
      frequency[token_to_id[token]] += 1

  def add_special_token(token, index):
    token_to_id[token] = index
    id_to_token.insert(index, token)

    # Set a very high frequency to avoid special tokens to be pruned.
    frequency.insert(index, float("inf"))

  add_special_token(constants.PADDING_TOKEN, constants.PADDING_ID)

  if args.with_sequence_tokens:
    add_special_token(constants.START_OF_SENTENCE_TOKEN, constants.START_OF_SENTENCE_ID)
    add_special_token(constants.END_OF_SENTENCE_TOKEN, constants.END_OF_SENTENCE_ID)

  tokenizer = getattr(tokenizers, args.tokenizer)()

  # Add each token from the corpus.
  with open(args.data, "rb") as data:
    for line in data:
      line = line.strip().decode("utf-8")
      tokens = tokenizer(line)
      for token in tokens:
        add_token(token)

  # Sort by frequency.
  sorted_ids = sorted(range(len(frequency)), key=lambda k: frequency[k], reverse=True)
  new_size = len(sorted_ids)

  # Discard words that do not meet frequency requirements.
  for i in range(new_size - 1, 0, -1):
    index = sorted_ids[i]
    if frequency[index] < args.min_frequency:
      new_size -= 1
    else:
      break

  # Limit absolute size.
  if args.size > 0:
    new_size = min(new_size, args.size)

  # Prune if needed.
  if new_size < len(id_to_token):
    new_id_to_token = []
    for i in range(new_size):
      index = sorted_ids[i]
      token = id_to_token[index]
      new_id_to_token.append(token)
    id_to_token = new_id_to_token

  # Generate the vocabulary file.
  with open(args.save_vocab, "wb") as vocab:
    for token in id_to_token:
      vocab.write(token.encode("utf-8"))
      vocab.write(b"\n")

if __name__ == "__main__":
  main()
