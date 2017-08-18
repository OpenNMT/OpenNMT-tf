"""Standalone script to generate word vocabularies from monolingual corpus."""

import argparse
import sys

from opennmt import constants

parser = argparse.ArgumentParser(description="Generate vocabulary file")
parser.add_argument(
  "data",
  help="Source text file.")
parser.add_argument(
  "--save_vocab", required=True,
  help="Output vocabulary file.")
parser.add_argument(
  "--delimiter", default=" ",
  help="Delimiter to split tokens. If empty, split each character.")
parser.add_argument(
  "--min_frequency", type=int, default=1,
  help="Minimum word frequency.")
parser.add_argument(
  "--size", type=int, default=0,
  help="Maximum vocabulary size. If = 0, do not limit vocabulary.")
parser.add_argument(
  "--with_sequence_tokens", type=bool, default=False,
  help="If True, also inject special sequence tokens (start, end).")
args = parser.parse_args()

token_to_id = {}
id_to_token = []
frequency = []

def add_token(token):
  if not token in token_to_id:
    id = len(id_to_token)
    token_to_id[token] = id
    id_to_token.append(token)
    frequency.append(1)
  else:
    frequency[token_to_id[token]] += 1

def add_special_token(token, id):
  token_to_id[token] = id
  id_to_token.insert(id, token)

  # Set a very high frequency to avoid special tokens to be pruned.
  frequency.insert(id, float("inf"))

add_special_token(constants.PADDING_TOKEN, constants.PADDING_ID)

if args.with_sequence_tokens:
  add_special_token(constants.START_OF_SENTENCE_TOKEN, constants.START_OF_SENTENCE_ID)
  add_special_token(constants.END_OF_SENTENCE_TOKEN, constants.END_OF_SENTENCE_ID)

# Add each token from the corpus.
with open(args.data, "rb") as data:
  for line in data:
    line = line.strip().decode("utf-8")
    if not args.delimiter:
      tokens = list(line)
    else:
      tokens = line.split(args.delimiter)
    for token in tokens:
      add_token(token)

# Sort by frequency.
sorted_ids = sorted(range(len(frequency)), key=lambda k: frequency[k], reverse=True)
new_size = len(sorted_ids)

# Discard words that do not meet frequency requirements.
for i in range(new_size - 1, 0, -1):
  id = sorted_ids[i]
  if frequency[id] < args.min_frequency:
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
    id = sorted_ids[i]
    token = id_to_token[id]
    new_id_to_token.append(token)
  id_to_token = new_id_to_token

# Generate the vocabulary file.
with open(args.save_vocab, "wb") as vocab:
  for token in id_to_token:
    vocab.write(token.encode("utf-8"))
    vocab.write(b"\n")
