"""Main library entrypoint."""

import copy
import io
import os
import sys
import random
import math
import subprocess
import time
import tempfile
import six
import yaml

import numpy as np
import tensorflow as tf

from opennmt import evaluation
from opennmt import models
from opennmt import training as training_util
from opennmt.data import dataset as dataset_util
from opennmt.utils import checkpoint as checkpoint_util
from opennmt.utils import misc


# These options require a value but we can fallback to a default one.
_CONFIG_FALLBACK = {
    "params": {},
    "train": {
        "batch_type": "examples",
        "length_bucket_width": 1,
        "sample_buffer_size": 500000,
        "save_summary_steps": 100
    },
    "eval": {
        "batch_size": 32
    },
    "infer": {
        "length_bucket_width": None,
        "batch_size": 16
    },
    "score": {
        "batch_size": 64
    }
}

class Runner(object):
  """Class for running and exporting models."""

  def __init__(self,
               model,
               config,
               auto_config=False,
               mixed_precision=False,
               seed=None):
    """Initializes the runner parameters.

    Args:
      model: A :class:`opennmt.models.Model` instance to run.
      config: The run configuration.
      auto_config: If ``True``, use automatic configuration values defined by
        :obj:`model`.
      mixed_precision: Enable mixed precision.
      seed: The random seed to set.
    """
    self._model = model
    self._optimizer = None
    self._config = config
    self._auto_config = auto_config
    self._mixed_precision = mixed_precision
    if mixed_precision:
      tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
      tf.random.set_seed(seed)

  def _finalize_config(self, training=False, num_devices=1):
    # Configuration priority: user config > auto config > default config.
    config = copy.deepcopy(_CONFIG_FALLBACK)
    if self._auto_config:
      model_config = self._model.auto_config(num_replicas=num_devices)
      if not model_config:
        raise NotImplementedError("This model does not define any automatic configuration values")
      misc.merge_dict(config, model_config)
    misc.merge_dict(config, self._config)
    config["params"].setdefault("num_hypotheses", config["infer"].get("n_best", 1))

    if training:
      train_config = config["train"]
      batch_size = train_config.get("batch_size")

      # Auto tune batch size.
      if batch_size is None or batch_size == 0:
        if train_config["batch_type"] == "examples":
          raise ValueError("Batch size autotuning is only supported for the \"tokens\" batch type")
        max_batch_size = 16384
        if train_config.get("effective_batch_size") is not None:
          max_batch_size = min(max_batch_size, train_config["effective_batch_size"])
        train_config["batch_size"] = _auto_tune_batch_size(
            config,
            max_batch_size=max_batch_size,
            num_devices=num_devices)

    tf.get_logger().info(
        "Using parameters:\n%s", yaml.dump(config, indent=2, default_flow_style=False))
    return config

  def _init_model(self, config):
    model = misc.clone_layer(self._model)
    model.initialize(config["data"], params=config["params"])
    if "optimizer" in config["params"]:
      optimizer = model.get_optimizer()
    else:
      optimizer = None
    checkpoint = checkpoint_util.Checkpoint(
        model,
        optimizer=optimizer,
        model_dir=config.get("model_dir"),
        keep_checkpoint_max=config["train"].get("keep_checkpoint_max", 8))
    return checkpoint

  def _init_run(self, training=False, num_devices=1):
    config = self._finalize_config(training=training, num_devices=num_devices)
    return self._init_model(config), config

  def train(self, num_devices=1, with_eval=False, checkpoint_path=None):
    """Runs the training loop.

    Args:
      num_devices: Number of devices to use for training.
      with_eval: Enable evaluation during training.
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      The path to the final model directory.
    """
    checkpoint, config = self._init_run(num_devices=num_devices, training=True)
    checkpoint.restore(
        checkpoint_path=checkpoint_path, weights_only=checkpoint_path is not None)

    model = checkpoint.model
    data_config = config["data"]
    train_config = config["train"]
    eval_config = config["eval"]

    batch_type = train_config["batch_type"]
    if batch_type == "tokens" and self._mixed_precision:
      batch_size_multiple = 8
    else:
      batch_size_multiple = 1

    dataset = model.examples_inputter.make_training_dataset(
        data_config["train_features_file"],
        data_config.get("train_labels_file"),
        train_config["batch_size"],
        batch_type=batch_type,
        batch_size_multiple=batch_size_multiple,
        shuffle_buffer_size=train_config["sample_buffer_size"],
        length_bucket_width=train_config["length_bucket_width"],
        maximum_features_length=train_config.get("maximum_features_length"),
        maximum_labels_length=train_config.get("maximum_labels_length"),
        single_pass=train_config.get("single_pass", False),
        prefetch_buffer_size=train_config.get("prefetch_buffer_size"))

    if with_eval:
      evaluator = evaluation.Evaluator.from_config(model, config)
    else:
      evaluator = None

    # Set gradients accumulation based on the requested effective batch size.
    if train_config.get("effective_batch_size") is not None:
      accum_steps = _count_batch_accum(
          train_config["batch_size"],
          train_config["effective_batch_size"],
          num_replicas=num_devices)
      tf.get_logger().info(
          "Accumulate gradients of %d iterations to reach effective batch size of %d",
          accum_steps,
          train_config["effective_batch_size"])
    else:
      accum_steps = 1

    trainer = training_util.Trainer(
        checkpoint,
        devices=misc.get_devices(count=num_devices),
        mixed_precision=self._mixed_precision)
    trainer(
        dataset,
        max_step=train_config.get("max_step"),
        accum_steps=accum_steps,
        report_steps=train_config.get("save_summary_steps", 100),
        save_steps=train_config.get("save_checkpoints_steps", 5000),
        evaluator=evaluator,
        eval_steps=eval_config.get("steps", 5000),
        export_on_best=eval_config.get("export_on_best"))
    average_last_checkpoints = train_config.get("average_last_checkpoints", 0)
    if average_last_checkpoints > 0:
      return self.average_checkpoints(
          os.path.join(checkpoint.model_dir, "avg"),
          max_count=average_last_checkpoints)
    return checkpoint.model_dir

  def evaluate(self, features_file=None, labels_file=None, checkpoint_path=None):
    """Runs evaluation.

    Args:
      features_file: The input features file to evaluate. If not set, will load
        ``eval_features_file`` from the data configuration.
      labels_file: The output labels file to evaluate. If not set, will load
        ``eval_labels_file`` from the data configuration.
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      A dict of evaluation metrics.
    """
    checkpoint, config = self._init_run()
    checkpoint_path = checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    step = int(checkpoint_path.split("-")[-1])
    evaluator = evaluation.Evaluator.from_config(
        checkpoint.model,
        config,
        features_file=features_file,
        labels_file=labels_file)
    return evaluator(step)

  def average_checkpoints(self, output_dir, max_count=8):
    """Averages checkpoints.

    Args:
      output_dir: The directory that will contain the averaged checkpoint.
      max_count: The maximum number of checkpoints to average.

    Returns:
      The path to the directory containing the averaged checkpoint.
    """
    checkpoint, _ = self._init_run()
    checkpoint.restore()
    model = checkpoint.model
    optimizer = checkpoint.optimizer
    model.create_variables(optimizer=optimizer)
    trackables = dict(model=model, optimizer=optimizer)
    return checkpoint_util.average_checkpoints(
        checkpoint.model_dir,
        output_dir,
        trackables,
        max_count=max_count)

  def update_vocab(self, output_dir, src_vocab=None, tgt_vocab=None):
    """Updates model vocabularies.

    Args:
      output_dir: Directory where the update checkpoint will be saved.
      src_vocab: Path to the new source vocabulary.
      tgt_vocab: Path to the new tagret vocabulary.

    Returns:
      Path to the new checkpoint directory.
    """
    if not isinstance(self._model, models.SequenceToSequence):
      raise ValueError("Updating vocabularies is only supported for sequence to sequence models")
    config = self._finalize_config()
    if src_vocab is None and tgt_vocab is None:
      return config["model_dir"]

    cur_checkpoint = self._init_model(config)
    cur_checkpoint.restore()
    model, optimizer = cur_checkpoint.model, cur_checkpoint.optimizer
    model.create_variables(optimizer=optimizer)

    new_config = copy.deepcopy(config)
    new_config["model_dir"] = output_dir
    if src_vocab is not None:
      new_config["data"]["source_vocabulary"] = src_vocab
    if tgt_vocab is not None:
      new_config["data"]["target_vocabulary"] = tgt_vocab
    new_checkpoint = self._init_model(new_config)
    new_model, new_optimizer = new_checkpoint.model, new_checkpoint.optimizer
    new_model.create_variables(optimizer=new_optimizer)

    model.transfer_weights(new_model, new_optimizer=new_optimizer, optimizer=optimizer)
    new_checkpoint.save(optimizer.iterations)
    return output_dir

  def infer(self,
            features_file,
            predictions_file=None,
            checkpoint_path=None,
            log_time=False):
    """Runs inference.

    Args:
      features_file: The file(s) to infer from.
      predictions_file: If set, predictions are saved in this file.
      checkpoint_path: Path of a specific checkpoint to predict. If ``None``,
        the latest is used.
      log_time: If ``True``, several time metrics will be printed in the logs at
        the end of the inference loop.
    """
    checkpoint, config = self._init_run()
    checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    model = checkpoint.model
    infer_config = config["infer"]
    dataset = model.examples_inputter.make_inference_dataset(
        features_file,
        infer_config["batch_size"],
        length_bucket_width=infer_config["length_bucket_width"],
        prefetch_buffer_size=infer_config.get("prefetch_buffer_size"))

    @dataset_util.function_on_next(dataset, as_numpy=True)
    def _predict(next_fn):
      source = next_fn()
      return model.infer(source)

    if predictions_file:
      stream = io.open(predictions_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    ordered_writer = None
    write_fn = lambda prediction: (
        model.print_prediction(prediction, params=infer_config, stream=stream))

    total_time = 0
    total_tokens = 0
    total_examples = 0
    start_time = time.time()

    for predictions in _predict():  # pylint: disable=no-value-for-parameter
      end_time = time.time()
      if log_time:
        total_time += end_time - start_time
        batch_size = next(six.itervalues(predictions)).shape[0]
        total_examples += batch_size
        length = predictions.get("length")
        if length is not None:
          if len(length.shape) == 2:
            length = length[:, 0]
          total_tokens += sum(length)
      for prediction in misc.extract_batches(predictions):
        if "index" in prediction:
          if ordered_writer is None:
            ordered_writer = misc.OrderRestorer(
                index_fn=lambda prediction: prediction["index"], callback_fn=write_fn)
          ordered_writer.push(prediction)
        else:
          write_fn(prediction)
      start_time = time.time()

    if log_time:
      tf.get_logger().info("Total prediction time (s): %f", total_time)
      tf.get_logger().info(
          "Average prediction time (s): %f", total_time / total_examples)
      if total_tokens > 0:
        tf.get_logger().info("Tokens per second: %f", total_tokens / total_time)
    if predictions_file:
      stream.close()

  def export(self, export_dir, checkpoint_path=None):
    """Exports a model.

    Args:
      export_dir: The export directory.
      checkpoint_path: The checkpoint path to export. If ``None``, the latest is used.
   """
    checkpoint, _ = self._init_run()
    checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    checkpoint.model.export(export_dir)

  def score(self, features_file, predictions_file, checkpoint_path=None, output_file=None):
    """Scores existing predictions.

    Args:
      features_file: The input file.
      predictions_file: The predictions file to score.
      checkpoint_path: Path of a specific checkpoint to use. If ``None``,
        the latest is used.
      output_file: The file where the scores are saved. Otherwise, they will be
        printed on the standard output.
    """
    checkpoint, config = self._init_run()
    checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    model = checkpoint.model
    score_config = config["score"]
    dataset = model.examples_inputter.make_evaluation_dataset(
        features_file,
        predictions_file,
        score_config["batch_size"],
        prefetch_buffer_size=score_config.get("prefetch_buffer_size"))

    @dataset_util.function_on_next(dataset, as_numpy=True)
    def _score(next_fn):
      features, labels = next_fn()
      return model.score(features, labels)

    if output_file:
      stream = io.open(output_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    for results in _score():  # pylint: disable=no-value-for-parameter
      for batch in misc.extract_batches(results):
        model.print_score(batch, params=score_config, stream=stream)

    if output_file:
      stream.close()


def _count_batch_accum(batch_size, target_batch_size, num_replicas=1):
  """Given the current batch size, the number of replicas, and the requested
  effective batch size, returns the number of gradients to accumulate.
  """
  return int(math.ceil(float(target_batch_size) / (batch_size * num_replicas)))

def _auto_tune_batch_size(config,
                          min_batch_size=1024,
                          max_batch_size=16384,
                          min_range=256,
                          sample_iterations=10,
                          num_devices=1,
                          scaling_factor=0.8):
  """Find the largest token-based batch size that can be used with this
  configuration.

  This function runs some training iterations and uses out-of-memory errors as
  search conditions. A binary search is used to converge to a suitable batch
  size.

  We prefer to run the iterations in a different process so that it does not
  alter the current context (OOM may not be safe to recover from, see for
  example https://stackoverflow.com/q/53820713/2529808).

  Args:
    config: The training configuration.
    min_batch_size: The smallest batch size to consider.
    max_batch_size: The largest batch size to consider.
    min_range: Continue searching while the difference between
      :obj:`max_batch_size` and :obj:`min_batch_size` is larger than this value.
    sample_iterations: The number of training iterations.
    num_devices: The number of devices to use.
    scaling_factor: Scale the found batch size by this value.

  Returns:
    The autotuned batch size.
  """
  model_dir = config["model_dir"]
  with tempfile.TemporaryDirectory() as tmpdir:
    config = copy.deepcopy(config)
    config["model_dir"] = tmpdir
    config["train"]["save_checkpoints_steps"] = None
    config["train"]["average_last_checkpoints"] = 0
    config["train"]["max_step"] = sample_iterations
    config_path = os.path.join(config["model_dir"], "batch_size_autotuner.yml")
    model_description = os.path.join(model_dir, "model_description.py")

    args = [
        "python", "-m", "opennmt.bin.main",
        "--config", config_path,
        "--model", model_description,
        "--checkpoint_path", model_dir,
        "train",
        "--num_gpus", str(num_devices)]

    tf.get_logger().info(
        "Searching the largest batch size between %d and %d with a precision of %d...",
        min_batch_size, max_batch_size, min_range)

    while max_batch_size - min_batch_size > min_range:
      batch_size = (max_batch_size + min_batch_size) // 2

      # Update configuration with current batch size and adjusted gradients
      # accumulation.
      config["train"]["batch_size"] = batch_size
      with tf.io.gfile.GFile(config_path, mode="wb") as config_file:
        yaml.dump(config, config_file)

      tf.get_logger().info("Trying training with batch size %d...", batch_size)
      time.sleep(1)
      with open(os.devnull, "w") as devnull:
        process = subprocess.Popen(args, stdout=devnull, stderr=devnull)
        exit_code = process.wait()

      if exit_code != 0:
        tf.get_logger().info("... failed.")
        max_batch_size = batch_size - 1
      else:
        tf.get_logger().info(
            "... succeeded, continue until the search range is smaller than %d.", min_range)
        min_batch_size = batch_size

  batch_size = int(scaling_factor * min_batch_size)
  tf.get_logger().info("Batch size auto tuned to %d.", batch_size)
  return batch_size
