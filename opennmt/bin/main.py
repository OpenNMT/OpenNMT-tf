"""Main script."""

import argparse
import json
import os
import six

import tensorflow as tf

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
  else:
    path = paths
    new_path = os.path.join(prefix, path)
    if os.path.isfile(new_path):
      return new_path
    else:
      return path

def main():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("run", choices=["train_and_eval", "train", "eval", "infer", "export"],
                      help="Run type.")
  parser.add_argument("--config", required=True, nargs="+",
                      help="List of configuration files.")
  parser.add_argument("--model_type", default="", choices=list(classes_in_module(catalog)),
                      help="Model type from the catalog.")
  parser.add_argument("--model", default="",
                      help="Custom model configuration file.")
  parser.add_argument("--run_dir", default="",
                      help="If set, model_dir will be created relative to this location.")
  parser.add_argument("--data_dir", default="",
                      help="If set, data files are expected to be relative to this location.")
  parser.add_argument("--features_file", default=[], nargs="+",
                      help="Run inference on this file.")
  parser.add_argument("--predictions_file", default="",
                      help=("File used to save predictions. If not set, predictions are printed "
                            "on the standard output."))
  parser.add_argument("--checkpoint_path", default=None,
                      help=("Checkpoint or directory to use for inference or export "
                            "(when a directory is set, the latest checkpoint is used)."))
  parser.add_argument("--num_gpus", type=int, default=1,
                      help="Number of GPUs to use for in-graph replication.")
  parser.add_argument("--chief_host", default="",
                      help="hostname:port of the chief worker (for distributed training).")
  parser.add_argument("--worker_hosts", default="",
                      help=("Comma-separated list of hostname:port of workers "
                            "(for distributed training)."))
  parser.add_argument("--ps_hosts", default="",
                      help=("Comma-separated list of hostname:port of parameter servers "
                            "(for distributed training)."))
  parser.add_argument("--task_type", default="chief",
                      choices=["chief", "worker", "ps", "evaluator"],
                      help="Type of the task to run (for distributed training).")
  parser.add_argument("--task_index", type=int, default=0,
                      help="ID of the task (for distributed training).")
  parser.add_argument("--log_level", default="INFO",
                      choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                      help="Logs verbosity.")
  parser.add_argument("--seed", type=int, default=None,
                      help="Random seed.")
  parser.add_argument("--gpu_allow_growth", default=False, action="store_true",
                      help="Allocate GPU memory dynamically.")
  args = parser.parse_args()

  tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

  # Setup cluster if defined.
  if args.chief_host:
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

  # Load and merge run configurations.
  config = load_config(args.config)
  if args.run_dir:
    config["model_dir"] = os.path.join(args.run_dir, config["model_dir"])
  if args.data_dir:
    config["data"] = _prefix_paths(args.data_dir, config["data"])

  if not os.path.isdir(config["model_dir"]):
    tf.logging.info("Creating model directory %s", config["model_dir"])
    os.makedirs(config["model_dir"])

  model = load_model(config["model_dir"], model_file=args.model, model_name=args.model_type)
  runner = Runner(
      model,
      config,
      seed=args.seed,
      num_devices=args.num_gpus,
      gpu_allow_growth=args.gpu_allow_growth)

  if args.run == "train_and_eval":
    runner.train_and_evaluate()
  elif args.run == "train":
    runner.train()
  elif args.run == "eval":
    runner.evaluate(checkpoint_path=args.checkpoint_path)
  elif args.run == "infer":
    if not args.features_file:
      parser.error("--features_file is required for inference.")
    elif len(args.features_file) == 1:
      args.features_file = args.features_file[0]
    runner.infer(
        args.features_file,
        predictions_file=args.predictions_file,
        checkpoint_path=args.checkpoint_path)
  elif args.run == "export":
    runner.export(checkpoint_path=args.checkpoint_path)


if __name__ == "__main__":
  main()
