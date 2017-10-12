"""Main script."""

import argparse
import json
import multiprocessing
import os
import pickle

import tensorflow as tf

from opennmt.utils import hooks
from opennmt.config import load_model_module, load_config


def setup_cluster(workers, parameter_servers, task_type, task_index):
  """Sets the cluster configuration.

  Args:
    workers: A list of worker hosts.
    parameter_servers: A list of parameter server hosts.
    task_type: The type of the local task.
    taks_index: The index of the local task.
  """
  # The master is the first worker.
  master = [workers.pop(0)]

  if task_type == "worker":
    if task_index == 0:
      task_type = "master"
    else:
      task_index -= 1

  cluster = {
      "ps": parameter_servers,
      "worker": workers,
      "master": master
  }

  os.environ["TF_CONFIG"] = json.dumps({
      "cluster": cluster,
      "task": {"type": task_type, "index": task_index},
      "environment": "cloud"
  })

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
  parser.add_argument("--ps_hosts", default="",
                      help="comma-separated list of hostname:port pairs")
  parser.add_argument("--worker_hosts", default="",
                      help="comma-separated list of hostname:port pairs")
  parser.add_argument("--task_type", default="worker", choices=["worker", "ps"],
                      help="type of the task to run")
  parser.add_argument("--task_index", type=int, default=0,
                      help="id of the task (0 is the master)")
  parser.add_argument("--log_level", default="INFO",
                      choices=["DEBUG", "ERROR", "FATAL", "INFO", "WARN"],
                      help="logs verbosity")
  parser.add_argument("--gpu_allow_growth", type=bool, default=False,
                      help="allocate gpu memory dynamically")
  args = parser.parse_args()

  tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

  # Setup cluster if defined.
  if args.worker_hosts:
    ps = args.ps_hosts.split(",")
    workers = args.workers_hosts.split(",")
    setup_cluster(workers, ps, args.task_type, args.task_index)

  # Load and merge run configurations.
  config = load_config(args.config)

  if not os.path.isdir(config["model_dir"]):
    tf.logging.info("Creating model directory %s", config["model_dir"])
    os.makedirs(config["model_dir"])

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = args.gpu_allow_growth

  run_config = tf.contrib.learn.RunConfig(
      model_dir=config["model_dir"],
      session_config=session_config)

  model = load_model(config["model_dir"], model_file=args.model)

  if args.run == "train":
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

    estimator = tf.estimator.Estimator(
        model,
        config=run_config,
        params=config["params"])

    batch_size = config["train"]["batch_size"]
    buffer_size = config["train"].get("buffer_size", batch_size * 1000)
    num_parallel_process_calls = config["train"].get(
        "num_parallel_process_calls", multiprocessing.cpu_count())
    num_buckets = config["train"].get("num_buckets", 5)
    maximum_features_length = config["train"].get("maximum_features_length", 0)
    maximum_labels_length = config["train"].get("maximum_labels_length", 0)

    train_input_fn = model.input_fn(
        tf.estimator.ModeKeys.TRAIN,
        batch_size,
        buffer_size,
        num_parallel_process_calls,
        config["data"],
        config["data"]["train_features_file"],
        labels_file=config["data"]["train_labels_file"],
        num_buckets=num_buckets,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length)
    eval_input_fn = model.input_fn(
        tf.estimator.ModeKeys.EVAL,
        batch_size,
        buffer_size,
        num_parallel_process_calls,
        config["data"],
        config["data"]["eval_features_file"],
        labels_file=config["data"]["eval_labels_file"],
        num_buckets=num_buckets)

    eval_hooks = []
    if config["train"].get("save_eval_predictions", False):
      save_path = os.path.join(config["model_dir"], "eval")
      if not os.path.isdir(save_path):
        os.makedirs(save_path)
      eval_hooks.append(hooks.SaveEvaluationPredictionHook(
          model, os.path.join(save_path, "predictions.txt")))

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=config["train"].get("train_steps"),
        eval_steps=None,
        eval_hooks=eval_hooks,
        min_eval_frequency=config["train"].get("eval_steps"),
        export_strategies=tf.contrib.learn.make_export_strategy(
            model.serving_input_fn(config["data"])))

    if args.task_type == "ps":
      experiment.run_std_server()
    elif run_config.is_chief:
      experiment.extend_train_hooks([
          hooks.LogParametersCountHook(),
          hooks.CountersHook(
              every_n_steps=run_config.save_summary_steps,
              output_dir=config["model_dir"])
      ])

      experiment.train_and_evaluate()
    else:
      experiment.train()
  elif args.run == "infer":
    if not args.features_file:
      parser.error("--features_file is required for inference.")

    estimator = tf.estimator.Estimator(
        model,
        config=run_config,
        params=config["params"])

    batch_size = config["infer"]["batch_size"]
    buffer_size = config["infer"].get("buffer_size", batch_size * 10)
    num_parallel_process_calls = config["infer"].get(
        "num_parallel_process_calls", multiprocessing.cpu_count())

    test_input_fn = model.input_fn(
        tf.estimator.ModeKeys.PREDICT,
        batch_size,
        buffer_size,
        num_parallel_process_calls,
        config["data"],
        args.features_file)

    for prediction in estimator.predict(input_fn=test_input_fn):
      model.print_prediction(prediction, params=config["infer"])
  elif args.run == "export":
    estimator = tf.estimator.Estimator(
        model,
        config=run_config,
        params=config["params"])

    estimator.export_savedmodel(
        os.path.join(config["model_dir"], "manual_export"),
        model.serving_input_fn(config["data"]))

if __name__ == "__main__":
  main()
