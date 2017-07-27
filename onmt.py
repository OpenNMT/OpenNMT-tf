import argparse
import yaml

from importlib import import_module

import tensorflow as tf
import opennmt as onmt

def load_config_module(path):
  """Loads a configuration file.

  Args:
    path: The relative path to the configuration file.

  Returns:
    A Python module.
  """
  module, _ = path.rsplit(".", 1)
  module = module.replace("/", ".")
  module = import_module(module)

  if not hasattr(module, "model"):
    raise ImportError("No model defined in " + path)

  return module

def main():
  parser = argparse.ArgumentParser(description="OpenNMT-tf.")
  parser.add_argument("--run", required=True,
                      help="run configuration file")
  parser.add_argument("--model", required=True,
                      help="model configuration file")
  args = parser.parse_args()

  # Load run configuration.
  params = {}
  run_config = tf.estimator.RunConfig()

  with open(args.run) as config_file:
    config = yaml.load(config_file.read())

    def _replace(run_config, section, key):
      kwargs = {}
      kwargs[key] = section[key]
      return run_config.replace(**kwargs)

    def _maybe_replace(run_config, section, key):
      if key in section:
        run_config = _replace(run_config, section, key)
      return run_config

    run_config = _replace(run_config, config["run"], "model_dir")
    run_config = _maybe_replace(run_config, config["run"], "save_checkpoints_steps")
    run_config = _maybe_replace(run_config, config["run"], "save_summary_steps")

    eval_every = config["run"].get("eval_steps")

    session_config = tf.ConfigProto()
    if "gpu_allow_growth" in config["run"]:
      session_config.gpu_options.allow_growth = config["run"]["gpu_allow_growth"]
    run_config = run_config.replace(session_config=session_config)

    if "params" in config:
      params.update(config["params"])
    params["log_dir"] = config["run"]["model_dir"]

  # Load model configuration.
  model_config = load_config_module(args.model)
  model = model_config.model()

  estimator = tf.estimator.Estimator(
    model_fn=model,
    config=run_config,
    params=params)

  buffer_size = config["data"].get("buffer_size") or 10000
  num_buckets = config["data"].get("num_buckets") or 5

  if config["run"]["type"] == "train":
    model_config.train(model)

    train_input_fn = model.input_fn(
      tf.estimator.ModeKeys.TRAIN,
      config["params"]["batch_size"],
      buffer_size,
      num_buckets,
      config["data"]["train_features_file"],
      labels_file=config["data"]["train_labels_file"])
    eval_input_fn = model.input_fn(
      tf.estimator.ModeKeys.EVAL,
      config["params"]["batch_size"],
      buffer_size,
      num_buckets,
      config["data"]["eval_features_file"],
      labels_file=config["data"]["eval_labels_file"])

    experiment = tf.contrib.learn.Experiment(
      estimator=estimator,
      train_input_fn=train_input_fn,
      eval_input_fn=eval_input_fn,
      train_steps_per_iteration=eval_every)

    experiment.continuous_train_and_eval()
  else:
    model_config.infer(model)

    test_input_fn = model.input_fn(
      tf.estimator.ModeKeys.PREDICT,
      config["params"]["batch_size"],
      buffer_size,
      num_buckets,
      config["data"]["features_file"],
      labels_file=config["data"].get("labels_file"))

    for predictions in estimator.predict(input_fn=test_input_fn):
      predictions = model.format_prediction(predictions, params=params)
      if not isinstance(predictions, list):
        predictions = [ predictions ]
      for prediction in predictions:
        print(prediction)

if __name__ == "__main__":
  main()
