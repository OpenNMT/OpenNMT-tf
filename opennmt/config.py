"""Defines functions related to configuration files."""

from importlib import import_module

import copy
import os
import sys
import tensorflow as tf
import yaml

from opennmt.models import catalog
from opennmt.optimizers import utils as optimizers_lib
from opennmt.schedules import lr_schedules as schedules_lib
from opennmt.utils.misc import merge_dict


MODEL_DESCRIPTION_FILENAME = "model_description.py"


def load_model_module(path):
    """Loads a model configuration file.

    Args:
      path: The relative path to the configuration file.

    Returns:
      A Python module.

    Raises:
      ValueError: if :obj:`path` is invalid.
      ImportError: if the module in :obj:`path` does not define a model.
    """
    if not os.path.exists(path):
        raise ValueError("Model configuration not found in %s" % path)
    dirname, filename = os.path.split(path)
    module_name, _ = os.path.splitext(filename)
    sys.path.insert(0, os.path.abspath(dirname))
    module = import_module(module_name)
    sys.path.pop(0)

    if not hasattr(module, "model"):
        raise ImportError("No model defined in {}".format(path))

    return module


def load_model_from_file(path, as_builder=False):
    """Loads a model from a configuration file.

    Args:
      path: The relative path to the configuration file.
      as_builder: If ``True``, return a callable building the model on call.

    Returns:
      A :class:`opennmt.models.Model` instance or a callable returning such
      instance.
    """
    module = load_model_module(path)
    model = module.model
    if not as_builder:
        model = model()
    del sys.path_importer_cache[os.path.dirname(module.__file__)]
    del sys.modules[module.__name__]
    return model


def load_model_from_catalog(name, as_builder=False):
    """Loads a model from the catalog.

    Args:
      name: The model name.
      as_builder: If ``True``, return a callable building the model on call.

    Returns:
      A :class:`opennmt.models.Model` instance or a callable returning such
      instance.

    Raises:
      ValueError: if the model :obj:`name` does not exist in the model catalog.
    """
    return catalog.get_model_from_catalog(name, as_builder=as_builder)


def load_model(
    model_dir, model_file=None, model_name=None, serialize_model=True, as_builder=False
):
    """Loads the model from the catalog or a definition file.

    Args:
      model_dir: The model directory.
      model_file: An optional model configuration.
        Mutually exclusive with :obj:`model_name`.
      model_name: An optional model name from the catalog.
        Mutually exclusive with :obj:`model_file`.
      serialize_model: Serialize the model definition in the model directory to
        make it optional for future runs.
      as_builder: If ``True``, return a callable building the model on call.

    Returns:
      A :class:`opennmt.models.Model` instance or a callable returning such
      instance.

    Raises:
      ValueError: if both :obj:`model_file` and :obj:`model_name` are set.
    """
    if model_file and model_name:
        raise ValueError("only one of model_file and model_name should be set")
    model_description_path = os.path.join(model_dir, MODEL_DESCRIPTION_FILENAME)

    if model_file:
        model = load_model_from_file(model_file, as_builder=as_builder)
        if serialize_model:
            tf.io.gfile.copy(model_file, model_description_path, overwrite=True)
    elif model_name:
        model = load_model_from_catalog(model_name, as_builder=as_builder)
        if serialize_model:
            with tf.io.gfile.GFile(
                model_description_path, mode="w"
            ) as model_description_file:
                model_description_file.write(
                    "from opennmt import models\n"
                    'model = lambda: models.get_model_from_catalog("%s")\n' % model_name
                )
    elif tf.io.gfile.exists(model_description_path):
        tf.get_logger().info(
            "Loading model description from %s", model_description_path
        )
        model = load_model_from_file(model_description_path, as_builder=as_builder)
    else:
        raise RuntimeError(
            "A model configuration is required: you probably need to "
            "set --model or --model_type on the command line."
        )

    return model


def load_config(config_paths, config=None):
    """Loads YAML configuration files.

    Args:
      config_paths: A list of configuration files that will be merged to a single
        configuration. The rightmost configuration takes priority.
      config: A (possibly non empty) config dictionary to fill.

    Returns:
      The configuration as Python dictionary.
    """
    if config is None:
        config = {}

    for config_path in config_paths:
        with tf.io.gfile.GFile(config_path) as config_file:
            subconfig = yaml.safe_load(config_file.read())
            # Add or update section in main configuration.
            merge_dict(config, subconfig)

    return config


def convert_to_v2_config(v1_config):
    """Converts a V1 configuration to its V2 equivalent.

    Args:
      v1_config: The V1 configuration.

    Returns:
      The V2 configuration.

    Raises:
      ValueError: if the conversion can not be done automatically.
    """
    config = copy.deepcopy(v1_config)

    _convert_to_v2_params(config)

    data_config = config.get("data")
    if data_config:
        # Covers most of seq2seq models in the catalog.
        _rename_opt(data_config, "source_words_vocabulary", "source_vocabulary")
        _rename_opt(data_config, "target_words_vocabulary", "target_vocabulary")

    for section_name in ("train", "eval", "infer", "score", "params"):
        section = config.get(section_name)
        if section is None:
            continue

        _delete_opt(section, "num_threads")
        _delete_opt(section, "prefetch_buffer_size")
        _rename_opt(section, "bucket_width", "length_bucket_width")
        if section_name == "train":
            _rename_opt(section, "train_steps", "max_step")
            _delete_opt(section, "save_checkpoints_secs")
        elif section_name == "eval":
            _delete_opt(section, "eval_delay")
            _delete_opt(section, "exporters")

        # Remove empty sections.
        if not section:
            config.pop(section_name)

    return config


def _convert_to_v2_params(config):
    params = config.get("params")
    if not params:
        return
    if "freeze_variables" in params:
        raise ValueError(
            "params/freeze_variables should be manually converted to "
            "params/freeze_layers"
        )

    _convert_to_v2_optimizer(params)
    _convert_to_v2_lr_schedules(params)
    _convert_to_v2_step_accumulation(params, config)
    _delete_opt(params, "param_init")
    _delete_opt(params, "loss_scale")
    _delete_opt(params, "horovod")
    _delete_opt(params, "maximum_learning_rate")
    _rename_opt(params, "maximum_iterations", "maximum_decoding_length")

    clip_gradients = _delete_opt(params, "clip_gradients")
    if clip_gradients:
        optimizer_params = params.setdefault("optimizer_params", {})
        optimizer_params["clipnorm"] = clip_gradients

    weight_decay = _delete_opt(params, "weight_decay")
    if weight_decay:
        optimizer_params = params.setdefault("optimizer_params", {})
        optimizer_params["weight_decay"] = weight_decay


# Only covering the most common optimizers.
_V1_OPTIMIZER_MAP = {
    "AdamOptimizer": "Adam",
    "GradientDescentOptimizer": "SGD",
    "LazyAdamOptimizer": "LazyAdam",
}


def _convert_to_v2_optimizer(params):
    optimizer = params.get("optimizer")
    if optimizer:
        try:
            # Check if the optimizer exists in V2.
            optimizers_lib.get_optimizer_class(optimizer)
        except ValueError as e:
            v2_optimizer = _V1_OPTIMIZER_MAP.get(optimizer)
            if not v2_optimizer:
                raise ValueError(
                    "params/optimizer should be manually converted: no registered "
                    "conversion for optimizer %s" % optimizer
                ) from e
            params["optimizer"] = v2_optimizer

    optimizer_params = params.get("optimizer_params")
    if optimizer_params:
        _rename_opt(optimizer_params, "beta1", "beta_1")
        _rename_opt(optimizer_params, "beta2", "beta_2")


def _convert_to_v2_lr_schedules(params):
    decay_type = params.get("decay_type")
    if not decay_type:
        return
    try:
        # Check if the learning rate schedule exists in V2.
        schedules_lib.get_lr_schedule_class(decay_type)
    except ValueError as e:
        if decay_type.startswith("noam_decay"):
            params["decay_type"] = "NoamDecay"
            if "decay_params" not in params:
                model_dim = _delete_opt(params, "decay_rate")
                warmup_steps = _delete_opt(params, "decay_steps")
                params["decay_params"] = dict(
                    model_dim=model_dim, warmup_steps=warmup_steps
                )
        else:
            raise ValueError(
                "params/decay_type should be manually converted: no registered "
                "conversion for decay type %s" % decay_type
            ) from e


def _convert_to_v2_step_accumulation(params, config):
    # Try to upgrade step accumulation to train/effective_batch_size.
    decay_step_duration = _delete_opt(params, "decay_step_duration") or 1
    gradients_accum = _delete_opt(params, "gradients_accum") or 1
    accum_steps = max(decay_step_duration, gradients_accum)
    if accum_steps > 1:
        train_config = config.setdefault("train", {})
        batch_size = train_config.get("batch_size")
        if batch_size is not None and batch_size > 0:
            train_config["effective_batch_size"] = batch_size * accum_steps
        else:
            raise ValueError(
                "params/decay_step_duration and params/gradients_accum "
                "should be manually converted to train/effective_batch_size"
            )


def _delete_opt(config, name):
    return config.pop(name, None)


def _rename_opt(config, name, new_name):
    value = _delete_opt(config, name)
    if value is not None:
        config[new_name] = value
    return value
