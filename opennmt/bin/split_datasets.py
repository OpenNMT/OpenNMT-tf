"""Standalone script to split multiple datasets into train, dev and test sets. For example:
python opennmt/bin/split_datasets.py data/toy-ende/src.txt data/toy-ende/tgt.txt 24774 --ratio 0.2
"""

import argparse
import random

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("files", type=str, nargs="+", help="File paths to split")
  parser.add_argument("size", type=int, help="Size of all the files in the above argument (needs to be the same)")
  parser.add_argument("--ratio", type=float, default=0.1, help="Ratio to use when making ")
  args = parser.parse_args()
  line_numbers = list(range(args.size))
  random.shuffle(line_numbers)
  split_size = int(args.ratio * args.size)
  dev_lines = set(line_numbers[:split_size])
  train_lines = set(line_numbers[split_size:-split_size])
  for filepath in args.files:
    with open(filepath) as data_file, open(filepath + '.train', 'w') as train_file, open(filepath + '.dev', 'w') as dev_file, open(filepath + '.test', 'w') as test_file:
      for line_num, line in enumerate(data_file):
        if line_num in train_lines:
          train_file.write(line)
        elif line_num in dev_lines:
          dev_file.write(line)
        else:
          test_file.write(line)


if __name__ == "__main__":
  main()