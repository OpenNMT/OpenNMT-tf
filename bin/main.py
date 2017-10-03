import argparse
import json
import os
import pickle
import sys

import tensorflow as tf

from opennmt.utils.misc import LogParametersCountHook, WordCounterHook
from opennmt.config import get_default_config, load_config_module, load_run_config


def setup_cluster(workers, ps, task_type, task_index):
  """Sets the cluster configuration.

  Args:
    workers: A list of worker hosts.
    ps: A list of parameter server hosts.
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
      "ps": ps,
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

    model_config = load_config_module(model_file)
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
  parser.add_argument("--run", required=True, nargs='+',
                      help="""list of run configuration files
                           (duplicate entries take the value of the rightmost file)""")
  parser.add_argument("--model", default="",
                      help="model configuration file")
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
  args = parser.parse_args()

  tf.logging.set_verbosity(getattr(tf.logging, args.log_level))

  # Load and merge run configurations.
  config = load_run_config(args.run, config=get_default_config())

  # Setup cluster if defined.
  if args.worker_hosts:
    ps = args.ps_hosts.split(",")
    workers = args.workers_hosts.split(",")
    setup_cluster(workers, ps, args.task_type, args.task_index)

  session_config = tf.ConfigProto()
  session_config.gpu_options.allow_growth = config["run"]["gpu_allow_growth"]

  run_config = tf.contrib.learn.RunConfig(
      save_summary_steps=config["run"]["save_summary_steps"],
      save_checkpoints_secs=None,
      save_checkpoints_steps=config["run"]["save_checkpoints_steps"],
      keep_checkpoint_max=config["run"]["keep_checkpoint_max"],
      log_step_count_steps=config["run"]["save_summary_steps"],
      model_dir=config["run"]["model_dir"],
      session_config=session_config)

  params = config["params"]
  params["log_dir"] = config["run"]["model_dir"]

  if not os.path.isdir(config["run"]["model_dir"]):
    tf.logging.info("Creating model directory %s", config["run"]["model_dir"])
    os.makedirs(config["run"]["model_dir"])

  model = load_model(config["run"]["model_dir"], model_file=args.model)

  estimator = tf.estimator.Estimator(
      model_fn=model,
      config=run_config,
      params=params)

  if config["run"]["type"] == "train":
    train_input_fn = model.input_fn(
        tf.estimator.ModeKeys.TRAIN,
        config["params"]["batch_size"],
        config["data"]["buffer_size"],
        config["data"]["num_threads"],
        config["data"]["num_buckets"],
        config["data"]["meta"],
        config["data"]["train_features_file"],
        labels_file=config["data"]["train_labels_file"],
        maximum_features_length=config["data"]["maximum_features_length"],
        maximum_labels_length=config["data"]["maximum_labels_length"])
    eval_input_fn = model.input_fn(
        tf.estimator.ModeKeys.EVAL,
        config["params"]["batch_size"],
        config["data"]["buffer_size"],
        config["data"]["num_threads"],
        config["data"]["num_buckets"],
        config["data"]["meta"],
        config["data"]["eval_features_file"],
        labels_file=config["data"]["eval_labels_file"])

    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fn=eval_input_fn,
        train_steps=config["run"]["train_steps"],
        eval_steps=None,
        min_eval_frequency=config["run"]["eval_steps"],
        export_strategies=tf.contrib.learn.make_export_strategy(
            model.serving_input_fn(config["data"]["meta"])))

    if args.task_type == "ps":
      experiment.run_std_server()
    elif run_config.is_chief:
      experiment.extend_train_hooks([
          LogParametersCountHook(),
          WordCounterHook(
              every_n_steps=config["run"]["save_summary_steps"],
              output_dir=config["run"]["model_dir"])
      ])

      experiment.train_and_evaluate()
    else:
      experiment.train()
  elif config["run"]["type"] == "infer":
    test_input_fn = model.input_fn(
        tf.estimator.ModeKeys.PREDICT,
        config["params"]["batch_size"],
        config["data"]["buffer_size"],
        config["data"]["num_threads"],
        config["data"]["num_buckets"],
        config["data"]["meta"],
        config["data"]["features_file"],
        labels_file=config["data"].get("labels_file"))

    for prediction in estimator.predict(input_fn=test_input_fn):
      model.print_prediction(prediction, params=params)
  elif config["run"]["type"] == "export":
    estimator.export_savedmodel(
        os.path.join(config["run"]["model_dir"], "manual_export"),
        model.serving_input_fn(config["data"]["meta"]))
  else:
    raise ValueError("Unknown run type: {}".format(config["run"]["type"]))

if __name__ == "__main__":
  main()
