"""Main script."""

import argparse
import json
import os
import six

import tensorflow as tf

from opennmt import __version__
from opennmt.models import catalog
from opennmt.runner import Runner
from opennmt.config import load_model, load_config
from opennmt.utils.misc import classes_in_module


def _prefix_paths(prefix, paths):
  """Recursively prefix paths.

  Args:
    prefix: The prefix to apply.
    data: A dict of relative paths.

  Returns:
    The updated dict.
  """
  if isinstance(paths, dict):
    for key, path in six.iteritems(paths):
      paths[key] = _prefix_paths(prefix, path)
    return paths
  elif isinstance(paths, list):
    for i, path in enumerate(paths):
      paths[i] = _prefix_paths(prefix, path)
    return paths
  else:
    path = paths
    new_path = os.path.join(prefix, path)
    if tf.io.gfile.exists(new_path):
      return new_path
    else:
      return path

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("-v", "--version", action="version", version="OpenNMT-tf %s" % __version__)
  parser.add_argument("--config", required=True, nargs="+",
                      help="List of configuration files.")
  parser.add_argument("--auto_config", default=False, action="store_true",
                      help="Enable automatic configuration values.")
  parser.add_argument("--model_type", default="",
                      choices=list(classes_in_module(catalog, public_only=True)),
                      help="Model type from the catalog.")
  parser.add_argument("--model", default="",
                      help="Custom model configuration file.")
  parser.add_argument("--run_dir", default="",
                      help="If set, model_dir will be created relative to this location.")
  parser.add_argument("--data_dir", default="",
                      help="If set, data files are expected to be relative to this location.")
  parser.add_argument("--checkpoint_path", default=None,
                      help=("Specific checkpoint or model directory to load "
                            "(when a directory is set, the latest checkpoint is used)."))
  parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of GPUs to use for in-graph replication.")
  parser.add_argument("--log_level", default="INFO",
                      choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                      help="Logs verbosity.")
  parser.add_argument("--seed", type=int, default=None,
                      help="Random seed.")
  parser.add_argument("--gpu_allow_growth", default=False, action="store_true",
                      help="Allocate GPU memory dynamically.")
  parser.add_argument("--intra_op_parallelism_threads", type=int, default=0,
                      help=("Number of intra op threads (0 means the system picks "
                            "an appropriate number)."))
  parser.add_argument("--inter_op_parallelism_threads", type=int, default=0,
                      help=("Number of inter op threads (0 means the system picks "
                            "an appropriate number)."))

  subparsers = parser.add_subparsers(help="Run type.", dest="run")
  parser_train = subparsers.add_parser("train", help="Training.")

  parser_train_and_eval = subparsers.add_parser("train_and_eval", help="Training and evaluation.")
  parser_train_and_eval.add_argument(
      "--chief_host", default="",
      help="hostname:port of the chief worker (for distributed training).")
  parser_train_and_eval.add_argument(
      "--worker_hosts", default="",
      help=("Comma-separated list of hostname:port of workers "
            "(for distributed training)."))
  parser_train_and_eval.add_argument(
      "--ps_hosts", default="",
      help=("Comma-separated list of hostname:port of parameter servers "
            "(for distributed training)."))
  parser_train_and_eval.add_argument(
      "--task_type", default="chief",
      choices=["chief", "worker", "ps", "evaluator"],
      help="Type of the task to run (for distributed training).")
  parser_train_and_eval.add_argument(
      "--task_index", type=int, default=0,
      help="ID of the task (for distributed training).")

  parser_eval = subparsers.add_parser("eval", help="Evaluation.")

  parser_infer = subparsers.add_parser("infer", help="Inference.")
  parser_infer.add_argument(
      "--features_file", nargs="+", required=True,
      help="Run inference on this file.")
  parser_infer.add_argument(
      "--predictions_file", default="",
      help=("File used to save predictions. If not set, predictions are printed "
            "on the standard output."))
  parser_infer.add_argument(
      "--log_prediction_time", default=False, action="store_true",
      help="Logs some prediction time metrics.")

  parser_export = subparsers.add_parser("export", help="Model export.")
  parser_export.add_argument(
      "--export_dir_base", default=None,
      help="The base directory of the exported model.")

  parser_score = subparsers.add_parser("score", help="Scoring.")
  parser_score.add_argument("--features_file", nargs="+", required=True,
                            help="Features file.")
  parser_score.add_argument("--predictions_file", default=None,
                            help="Predictions to score.")

  parser_average_checkpoints = subparsers.add_parser(
      "average_checkpoints", help="Checkpoint averaging.")
  parser_average_checkpoints.add_argument(
      "--output_dir", required=True,
      help="The output directory for the averaged checkpoint.")
  parser_average_checkpoints.add_argument(
      "--max_count", type=int, default=8,
      help="The maximal number of checkpoints to average.")

  args = parser.parse_args()

  tf.compat.v1.logging.set_verbosity(getattr(tf.compat.v1.logging, args.log_level))
  tf.config.threading.set_intra_op_parallelism_threads(args.intra_op_parallelism_threads)
  tf.config.threading.set_inter_op_parallelism_threads(args.inter_op_parallelism_threads)
  if args.gpu_allow_growth:
    for device in tf.config.experimental.list_physical_devices(device_type="GPU"):
      tf.config.experimental.set_memory_growth(device, enable=True)

  # Setup cluster if defined.
  if args.run == "train_and_eval" and args.chief_host:
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "chief": [args.chief_host],
            "worker": args.worker_hosts.split(","),
            "ps": args.ps_hosts.split(",")
        },
        "task": {
            "type": args.task_type,
            "index": args.task_index
        }
    })
    is_chief = args.task_type == "chief"
  else:
    is_chief = True

  # Load and merge run configurations.
  config = load_config(args.config)
  if args.run_dir:
    config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
  if args.data_dir:
    config["data"] = _prefix_paths(args.data_dir, config["data"])

  if is_chief and not tf.io.gfile.exists(config["model_dir"]):
    tf.compat.v1.logging.info("Creating model directory %s", config["model_dir"])
    tf.io.gfile.makedirs(config["model_dir"])

  model = load_model(
      config["model_dir"],
      model_file=args.model,
      model_name=args.model_type,
      serialize_model=is_chief)
  runner = Runner(
      model,
      config,
      seed=args.seed,
      num_devices=args.num_gpus,
      auto_config=args.auto_config)

  if args.run == "train_and_eval":
    runner.train(checkpoint_path=args.checkpoint_path, with_eval=True)
  elif args.run == "train":
    runner.train(checkpoint_path=args.checkpoint_path)
  elif args.run == "eval":
    runner.evaluate(checkpoint_path=args.checkpoint_path)
  elif args.run == "infer":
    if len(args.features_file) == 1:
      args.features_file = args.features_file[0]
    runner.infer(
        args.features_file,
        predictions_file=args.predictions_file,
        checkpoint_path=args.checkpoint_path,
        log_time=args.log_prediction_time)
  elif args.run == "export":
    runner.export(
        checkpoint_path=args.checkpoint_path,
        export_dir_base=args.export_dir_base)
  elif args.run == "score":
    runner.score(
        args.features_file,
        args.predictions_file,
        checkpoint_path=args.checkpoint_path)
  elif args.run == "average_checkpoints":
    runner.average_checkpoints(args.output_dir, max_count=args.max_count)


if __name__ == "__main__":
  main()
