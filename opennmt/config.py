"""Defines functions related to configuration files."""

from importlib import import_module

import os
import pickle
import sys
import tensorflow as tf
import yaml

from opennmt.models import catalog
from opennmt.utils import compat
from opennmt.utils.misc import merge_dict


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
  sys.path.pop(0)

  if not hasattr(module, "model"):
    raise ImportError("No model defined in {}".format(path))

  return module

def load_model_from_file(path):
  """Loads a model from a configuration file.

  Args:
    path: The relative path to the configuration file.

  Returns:
    A :class:`opennmt.models.model.Model` instance.
  """
  module = load_model_module(path)
  model = module.model()
  del sys.path_importer_cache[os.path.dirname(module.__file__)]
  del sys.modules[module.__name__]
  return model

def load_model_from_catalog(name):
  """Loads a model from the catalog.

  Args:
    name: The model name.

  Returns:
    A :class:`opennmt.models.model.Model` instance.
  """
  return getattr(catalog, name)()

def load_model(model_dir,
               model_file=None,
               model_name=None,
               serialize_model=True):
  """Loads the model from the catalog or a file.

  The model object is pickled in :obj:`model_dir` to make the model
  configuration optional for future runs.

  Args:
    model_dir: The model directory.
    model_file: An optional model configuration.
      Mutually exclusive with :obj:`model_name`.
    model_name: An optional model name from the catalog.
      Mutually exclusive with :obj:`model_file`.
    serialize_model: Serialize the model definition in the model directory.

  Returns:
    A :class:`opennmt.models.model.Model` instance.

  Raises:
    ValueError: if both :obj:`model_file` and :obj:`model_name` are set.
  """
  if model_file and model_name:
    raise ValueError("only one of model_file and model_name should be set")
  model_name_or_path = model_file or model_name
  model_description_path = os.path.join(model_dir, "model_description.py")

  # Also try to load the pickled model for backward compatibility.
  serial_model_file = os.path.join(model_dir, "model_description.pkl")

  if model_name_or_path:
    if tf.train.latest_checkpoint(model_dir) is not None:
      compat.logging.warn(
          "You provided a model configuration but a checkpoint already exists. "
          "The model configuration must define the same model as the one used for "
          "the initial training. However, you can change non structural values like "
          "dropout.")

    if model_file:
      model = load_model_from_file(model_file)
      if serialize_model:
        compat.gfile_copy(model_file, model_description_path, overwrite=True)
    elif model_name:
      model = load_model_from_catalog(model_name)
      if serialize_model:
        with compat.gfile_open(model_description_path, mode="w") as model_description_file:
          model_description_file.write("from opennmt.models import catalog\n")
          model_description_file.write("model = catalog.%s\n" % model_name)
  elif compat.gfile_exists(model_description_path):
    compat.logging.info("Loading model description from %s", model_description_path)
    model = load_model_from_file(model_description_path)
  elif compat.gfile_exists(serial_model_file):
    compat.logging.info("Loading serialized model description from %s", serial_model_file)
    with compat.gfile_open(serial_model_file, mode="rb") as serial_model:
      model = pickle.load(serial_model)
  else:
    raise RuntimeError("A model configuration is required: you probably need to "
                       "set --model or --model_type on the command line.")

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
    with compat.gfile_open(config_path, mode="rb") as config_file:
      subconfig = yaml.load(config_file.read())
      # Add or update section in main configuration.
      merge_dict(config, subconfig)

  return config
