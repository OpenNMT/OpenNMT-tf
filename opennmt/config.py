"""Defines functions related to configuration files."""

from importlib import import_module

import yaml


def load_model_module(path):
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
    raise ImportError("No model defined in {}".format(path))

  return module

def load_config(config_paths, config=None):
  """Loads configuration files.

  Args:
    config_paths: A list of configuration files.
    config: A (possibly non empty) config dictionary to fill.

  Returns:
    The configuration dictionary.
  """
  if config is None:
    config = {}

  for config_path in config_paths:
    with open(config_path) as config_file:
      subconfig = yaml.load(config_file.read())

      # Add or update section in main configuration.
      for section in subconfig:
        if section in config:
          if isinstance(config[section], dict):
            config[section].update(subconfig[section])
          else:
            config[section] = subconfig[section]
        else:
          config[section] = subconfig[section]

  return config
