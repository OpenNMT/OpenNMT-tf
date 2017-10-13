"""Main script."""

import argparse
import json
import multiprocessing
import os
import pickle

import tensorflow as tf

from opennmt.utils import hooks
from opennmt.config import load_model_module, load_config


def load_model(model_dir, model_file=None):
  """Loads the model.

  The model object is pickled in `model_dir` to make the model configuration
  optional for future runs.

  Args:
    model_dir: The model directory.
    model_file: An optional model configuration.

  Returns:
    A `opennmt.models.Model` object.
  """
  serial_model_file = os.path.join(model_dir, "model_description.pkl")

  if model_file:
    if tf.train.checkpoint_exists(model_dir):
      tf.logging.warn(
          "You provided a model configuration but a checkpoint already exists. "
          "The model configuration must match the one used for the initial training.")

    model_config = load_model_module(model_file)
    model = model_config.model()

    with open(serial_model_file, "wb") as serial_model:
      pickle.dump(model, serial_model)
  elif not os.path.isfile(serial_model_file):
    raise RuntimeError("A model configuration is required.")
  else:
    tf.logging.info("Loading serialized model description from %s", serial_model_file)
    with open(serial_model_file, "rb") as serial_model:
      model = pickle.load(serial_model)

  return model

def train(estimator, model, config):
  """Runs training.

  Args:
    estimator: A `tf.estimator.Estimator`.
    model: A `opennmt.models.Model`.
    config: The configuration.
  """
  batch_size = config["train"]["batch_size"]
  buffer_size = config["train"].get("buffer_size", batch_size * 1000)
  num_parallel_process_calls = config["train"].get(
      "num_parallel_process_calls", multiprocessing.cpu_count())

  train_hooks = [
      hooks.LogParametersCountHook(),
      hooks.CountersHook(
          every_n_steps=estimator.config.save_summary_steps,
          output_dir=estimator.model_dir)]

  eval_hooks = []
  if config["train"].get("save_eval_predictions", False):
    save_path = os.path.join(estimator.model_dir, "eval")
    if not os.path.isdir(save_path):
      os.makedirs(save_path)
    eval_hooks.append(hooks.SaveEvaluationPredictionHook(
        model, os.path.join(save_path, "predictions.txt")))

  train_spec = tf.estimator.TrainSpec(
      input_fn=model.input_fn(
          tf.estimator.ModeKeys.TRAIN,
          batch_size,
          buffer_size,
          num_parallel_process_calls,
          config["data"],
          config["data"]["train_features_file"],
          labels_file=config["data"]["train_labels_file"],
          num_buckets=config["train"].get("num_buckets", 5),
          maximum_features_length=config["train"].get("maximum_features_length", 0),
          maximum_labels_length=config["train"].get("maximum_labels_length", 0)),
      max_steps=config["train"].get("train_steps"),
      hooks=train_hooks)

  eval_spec = tf.estimator.EvalSpec(
      input_fn=model.input_fn(
          tf.estimator.ModeKeys.EVAL,
          batch_size,
          buffer_size,
          num_parallel_process_calls,
          config["data"],
          config["data"]["eval_features_file"],
          labels_file=config["data"]["eval_labels_file"]),
      steps=None,
      hooks=eval_hooks,
      exporters=tf.estimator.LatestExporter("latest", model.serving_input_fn(config["data"])),
      start_delay_secs=config["train"].get("eval_delay", 3600))

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def infer(features_file, estimator, model, config, checkpoint_path=None):
  """Runs inference and prints predictions on the standard output.

  Args:
    features_file: The file to infer from.
    estimator: A `tf.estimator.Estimator`.
    model: A `opennmt.models.Model`.
    config: The configuration.
    checkpoint_path: Path of a specific checkpoint to predict. If `None`, the
      latest is used.
  """
  if "infer" not in config:
    config["infer"] = {}

  batch_size = config["infer"].get("batch_size", 1)
  input_fn = model.input_fn(
      tf.estimator.ModeKeys.PREDICT,
      batch_size,
      config["infer"].get("buffer_size", batch_size * 10),
      config["infer"].get("num_parallel_process_calls", multiprocessing.cpu_count()),
      config["data"],
      features_file)

  for prediction in estimator.predict(input_fn=input_fn, checkpoint_path=checkpoint_path):
    model.print_prediction(prediction, params=config["infer"])

def export(estimator, model, config, checkpoint_path=None):
  """Exports a model.

  Args:
    estimator: A `tf.estimator.Estimator`.
    model: A `opennmt.models.Model`.
    config: The configuration.
    checkpoint_path: The checkpoint path to export. If `None`, the latest is used.
  """
  export_dir = os.path.join(estimator.model_dir, "export")
  if not os.path.isdir(export_dir):
    os.makedirs(export_dir)

  estimator.export_savedmodel(
      os.path.join(export_dir, "manual"),
      model.serving_input_fn(config["data"]),
      checkpoint_path=checkpoint_path)


def main():
  parser = argparse.ArgumentParser(description="OpenNMT-tf.")
  parser.add_argument("run", choices=["train", "infer", "export"],
                      help="run type")
  parser.add_argument("--config", required=True, nargs="+",
                      help="""list of configuration files""")
  parser.add_argument("--model", default="",
                      help="model configuration file")
  parser.add_argument("--features_file", default="",
                      help="run inference on this file")
  parser.add_argument("--checkpoint_path", default=None,
                      help="checkpoint to use for inference or export (latest by default)")
  parser.add_argument("--chief_host", default="",
                      help="hostname:port of the chief worker")
  parser.add_argument("--worker_hosts", default="",
                      help="comma-separated list of hostname:port of workers")
  parser.add_argument("--ps_hosts", default="",
                      help="comma-separated list of hostname:port of parameter servers")
  parser.add_argument("--task_type", default="chief",
                      choices=["chief", "worker", "ps", "evaluator"],
                      help="type of the task to run")
  parser.add_argument("--task_index", type=int, default=0,
                      help="id of the task")
  parser.add_argument("--log_level", default="INFO",
                      choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                      help="logs verbosity")
  parser.add_argument("--gpu_allow_growth", type=bool, default=False,
                      help="allocate gpu memory dynamically")
  args = parser.parse_args()

  tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

  # Setup cluster if defined.
  if args.chief_host:
    os.environ["TF_CONFIG"] = json.dumps({
        "cluster": {
            "chief": args.chief_host,
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

  if not os.path.isdir(config["model_dir"]):
    tf.logging.info("Creating model directory %s", config["model_dir"])
    os.makedirs(config["model_dir"])

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = args.gpu_allow_growth

  run_config = tf.estimator.RunConfig(
      model_dir=config["model_dir"],
      session_config=session_config)

  if "save_summary_steps" in config["train"]:
    run_config = run_config.replace(
        save_summary_steps=config["train"]["save_summary_steps"],
        log_step_count_steps=config["train"]["save_summary_steps"])
  if "save_checkpoints_steps" in config["train"]:
    run_config = run_config.replace(
        save_checkpoints_secs=None,
        save_checkpoints_steps=config["train"]["save_checkpoints_steps"])
  if "keep_checkpoint_max" in config["train"]:
    run_config = run_config.replace(
        keep_checkpoint_max=config["train"]["keep_checkpoint_max"])

  model = load_model(config["model_dir"], model_file=args.model)

  estimator = tf.estimator.Estimator(
      model,
      config=run_config,
      params=config["params"])

  if args.run == "train":
    train(estimator, model, config)
  elif args.run == "infer":
    if not args.features_file:
      parser.error("--features_file is required for inference.")
    infer(args.features_file, estimator, model, config, checkpoint_path=args.checkpoint_path)
  elif args.run == "export":
    export(estimator, model, config, checkpoint_path=args.checkpoint_path)


if __name__ == "__main__":
  main()
