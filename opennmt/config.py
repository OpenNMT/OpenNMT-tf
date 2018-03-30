"""Defines functions related to configuration files."""

from importlib import import_module

import io
import os
import pickle
import sys
import yaml

import tensorflow as tf


def load_model_module(path):
  """Loads a model configuration file.

  Args:
    path: The relative path to the configuration file.

  Returns:
    A Python module.
  """
  dirname, filename = os.path.split(path)
  module_name, _ = os.path.splitext(filename)
  sys.path.insert(0, os.path.abspath(dirname))
  module = import_module(module_name)

  if not hasattr(module, "model"):
    raise ImportError("No model defined in {}".format(path))

  return module

def load_model(model_dir, model_file=None):
  """Loads the model.

  The model object is pickled in :obj:`model_dir` to make the model
  configuration optional for future runs.

  Args:
    model_dir: The model directory.
    model_file: An optional model configuration.

  Returns:
    A :class:`opennmt.models.model.Model` object.
  """
  serial_model_file = os.path.join(model_dir, "model_description.pkl")

  if model_file:
    if tf.train.latest_checkpoint(model_dir) is not None:
      tf.logging.warn(
          "You provided a model configuration but a checkpoint already exists. "
          "The model configuration must define the same model as the one used for "
          "the initial training. However, you can change non structural values like "
          "dropout.")

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
    with io.open(config_path, encoding="utf-8") as config_file:
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
