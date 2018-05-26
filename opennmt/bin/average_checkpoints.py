"""Checkpoint averaging script."""

import argparse

import tensorflow as tf

from opennmt.utils.checkpoint import average_checkpoints


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model_dir", required=True,
                      help="The model directory containing the checkpoints.")
  parser.add_argument("--output_dir", required=True,
                      help="The output directory where the averaged checkpoint will be saved.")
  parser.add_argument("--max_count", type=int, default=8,
                      help="The maximal number of checkpoints to average.")
  args = parser.parse_args()
  average_checkpoints(args.model_dir, args.output_dir, max_count=args.max_count)


if __name__ == "__main__":
  main()
