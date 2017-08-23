"""Defines functions related to configuration files."""

import yaml

from importlib import import_module


def get_default_config():
  """Returns a default run configuration.

  Required options are not set but expected to be provided by
  the user.

  Returns:
    A dictionary with the same structure as calling `yaml.load`
    on the YAML configuration file.
  """
  return {
    "run": {
      "save_checkpoints_steps": 1000,
      "keep_checkpoint_max": 5,
      "save_summary_steps": 100,
      "eval_steps": None,
      "gpu_allow_growth": False
    },
    "data": {
      "maximum_features_length": 0,
      "maximum_labels_length": 0,
      "buffer_size": 10000,
      "num_buckets": 5
    },
    "params": {
      "clip_gradients": None,
      "decay_type": None,
      "staircase": True,
      "start_decay_steps": 0,
      "minimum_learning_rate": 0,
      "beam_width": 5,
      "length_penalty": 0.2,
      "maximum_iterations": 250,
      "n_best": 1
    }
  }

def load_config_module(path):
  """Loads a model configuration file.

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

def load_run_config(run_files, config={}):
  """Loads run configuration files.

  Args:
    run_files: A list of run configuration files.
    config: A (possibly non empty) config dictionary to fill.

  Returns:
    The configuration dictionary.
  """
  for config_path in run_files:
    with open(config_path) as config_file:
      subconfig = yaml.load(config_file.read())

      # Add or update section in main configuration.
      for section in subconfig:
        if section in config:
          config[section].update(subconfig[section])
        else:
          config[section] = subconfig[section]

  return config
