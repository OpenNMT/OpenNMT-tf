"""Script to convert checkpoint variables from one data type to another."""

import argparse

import tensorflow as tf

from opennmt.utils import checkpoint


def main():
  tf.logging.set_verbosity(tf.logging.INFO)

  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--model_dir", default=None,
                      help="The path to the model directory.")
  parser.add_argument("--checkpoint_path", default=None,
                      help="The path to the checkpoint to convert.")
  parser.add_argument("--output_dir", required=True,
                      help="The output directory where the updated checkpoint will be saved.")
  parser.add_argument("--target_dtype", required=True,
                      help="Target data type (e.g. float16 or float32).")
  parser.add_argument("--source_dtype", default=None,
                      help="Source data type (e.g. float16 or float32, inferred if not set).")
  args = parser.parse_args()
  if args.model_dir is None and args.checkpoint_path is None:
    raise ValueError("One of --checkpoint_path and --model_dir should be set")
  checkpoint_path = args.checkpoint_path
  if checkpoint_path is None:
    checkpoint_path = tf.train.latest_checkpoint(args.model_dir)
  target_dtype = tf.as_dtype(args.target_dtype)
  if args.source_dtype is None:
    source_dtype = tf.float32 if target_dtype == tf.float16 else tf.float16
  else:
    source_dtype = tf.as_dtype(args.source_dtype)
  checkpoint.convert_checkpoint(
      checkpoint_path,
      args.output_dir,
      source_dtype,
      target_dtype,
      session_config=tf.ConfigProto(device_count={"GPU": 0}))


if __name__ == "__main__":
  main()
