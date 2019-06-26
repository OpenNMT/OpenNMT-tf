"""Main library entrypoint."""

import copy
import io
import os
import sys
import random
import math
import subprocess
import json
import yaml
import time
import six
import itertools
import collections

import numpy as np
import tensorflow as tf

from google.protobuf import text_format

from opennmt import evaluation
from opennmt import models
from opennmt import training
from opennmt.data import dataset as dataset_util
from opennmt.utils import checkpoint, misc


# These options require a value but we can fallback to a default one.
_CONFIG_FALLBACK = {
    "params": {},
    "train": {
        "batch_type": "examples",
        "bucket_width": 1,
        "sample_buffer_size": 500000,
        "save_summary_steps": 100
    },
    "eval": {
        "batch_size": 32,
        "eval_delay": 18000,
        "exporters": "last"
    },
    "infer": {
        "bucket_width": None,
        "batch_size": 16
    },
    "score": {
        "batch_size": 64
    }
}

class Runner(object):
  """Class for managing training, inference, and export."""

  def __init__(self,
               model,
               config,
               seed=None,
               num_devices=1,
               auto_config=False):
    """Initializes the runner parameters.

    Args:
      model: A :class:`opennmt.models.model.Model` instance to run.
      config: The run configuration.
      seed: The random seed to set.
      num_devices: The number of devices (GPUs) to use for training.
      auto_config: If ``True``, use automatic configuration values defined by
        :obj:`model`.

    Raises:
      NotImplementedError: If :obj:`auto_config` is ``True`` but :obj:`model`
        does not define any automatic configuration values.
    """
    self._model = model
    self._num_devices = num_devices
    self._num_replicas = num_devices
    self._seed = seed

    # Configuration priority: user config > auto config > default config.
    self._config = copy.deepcopy(_CONFIG_FALLBACK)
    if auto_config:
      model_config = self._model.auto_config(num_replicas=self._num_replicas)
      if not model_config:
        raise NotImplementedError("This model does not define any automatic configuration values")
      misc.merge_dict(self._config, model_config)
    misc.merge_dict(self._config, config)
    tf.get_logger().info(
        "Using parameters:\n%s", yaml.dump(self._config, indent=2, default_flow_style=False))

    self._config["params"].setdefault("num_hypotheses", self._config["infer"].get("n_best", 1))
    self._model.initialize(self._config["data"], params=self._config["params"])
    if "optimizer" in self._config["params"]:
      self._optimizer = self._model.get_optimizer()
    else:
      self._optimizer = None
    self._checkpoint = checkpoint.Checkpoint(
        self._model,
        optimizer=self._optimizer,
        model_dir=self._config.get("model_dir"),
        keep_checkpoint_max=self._config["train"].get("keep_checkpoint_max", 8))

    if seed is not None:
      np.random.seed(seed)
      random.seed(seed)
      tf.random.set_seed(seed)

  def is_chief(self):
    """Returns ``True`` if this runner is the master runner."""
    cluster_spec = os.getenv("TF_CONFIG")
    if cluster_spec is None:
      return True
    cluster_spec = json.loads(cluster_spec)
    return cluster_spec["task"]["type"] == "chief"

  def _finalize_training_parameters(self):
    train_config = self._config["train"]
    batch_size = train_config.get("batch_size")

    # Auto tune batch size.
    if batch_size is None or batch_size == 0:
      if train_config["batch_type"] == "examples":
        raise ValueError("Batch size autotuning is only supported for the \"tokens\" batch type")
      max_batch_size = 16384
      if train_config.get("effective_batch_size") is not None:
        max_batch_size = min(max_batch_size, train_config["effective_batch_size"])
      train_config["batch_size"] = _auto_tune_batch_size(
          self._config,
          max_batch_size=max_batch_size,
          num_devices=self._num_devices)

    # Set gradients accumulation based on the requested effective batch size.
    if train_config.get("effective_batch_size") is not None:
      self._config["params"]["gradients_accum"] = _count_batch_accum(
          train_config["batch_size"],
          train_config["effective_batch_size"],
          num_replicas=self._num_replicas)
      tf.get_logger().info(
          "Accumulate gradients of %d iterations to reach effective batch size of %d",
          self._config["params"]["gradients_accum"],
          train_config["effective_batch_size"])

  def train(self, checkpoint_path=None, with_eval=False):
    """Runs the training loop.

    Args:
      checkpoint_path: The checkpoint path to load the model weights from it.
      with_eval: Enable evaluation during training.

    Returns:
      The path to the final model directory.
    """
    self._finalize_training_parameters()
    self._checkpoint.restore(
        checkpoint_path=checkpoint_path, weights_only=checkpoint_path is not None)
    params = self._config["params"]
    data_config = self._config["data"]
    train_config = self._config["train"]
    eval_config = self._config["eval"]
    dataset = self._model.examples_inputter.make_training_dataset(
        data_config["train_features_file"],
        data_config.get("train_labels_file"),
        train_config["batch_size"],
        batch_type=train_config["batch_type"],
        shuffle_buffer_size=train_config["sample_buffer_size"],
        bucket_width=train_config["bucket_width"],
        maximum_features_length=train_config.get("maximum_features_length"),
        maximum_labels_length=train_config.get("maximum_labels_length"),
        single_pass=train_config.get("single_pass", False),
        num_threads=train_config.get("num_threads", 4),
        prefetch_buffer_size=train_config.get("prefetch_buffer_size"))

    if with_eval:
      evaluator = evaluation.Evaluator.from_config(self._model, self._config)
    else:
      evaluator = None
    trainer = training.Trainer(self._checkpoint)
    trainer(
        dataset,
        max_step=train_config.get("train_steps"),
        accum_steps=params.get("gradients_accum", 1),
        report_steps=train_config.get("save_summary_steps", 100),
        save_steps=train_config.get("save_checkpoints_steps", 5000),
        evaluator=evaluator,
        eval_steps=eval_config.get("steps", 5000))
    return self._maybe_average_checkpoints()

  def evaluate(self, checkpoint_path=None):
    """Runs evaluation.

    Args:
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      A dict of evaluation metrics.
    """
    checkpoint_path = self._checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    step = int(checkpoint_path.split("-")[-1])
    evaluator = evaluation.Evaluator.from_config(self._model, self._config)
    return evaluator(step)

  def _maybe_average_checkpoints(self, avg_subdirectory="avg"):
    """Averages checkpoints if enabled in the training configuration and if the
    current training instance is the chief.

    Args:
      avg_subdirectory: The directory within the model directory that will
        contain the averaged checkpoint.

    Returns:
      The path to the latest model directory.
    """
    average_last_checkpoints = self._config["train"].get("average_last_checkpoints", 0)
    model_dir = self._config["model_dir"]
    if average_last_checkpoints > 0 and self.is_chief():
      return self.average_checkpoints(
          os.path.join(model_dir, avg_subdirectory),
          max_count=average_last_checkpoints)
    return model_dir

  def average_checkpoints(self, output_dir, max_count=8):
    """Averages checkpoints.

    Args:
      output_dir: The directory that will contain the averaged checkpoint.
      max_count: The maximum number of checkpoints to average.

    Returns:
      The path to the directory containing the averaged checkpoint.
    """
    # Create all variables.
    self._model.create_variables(optimizer=self._optimizer)
    trackables = dict(model=self._model, optimizer=self._optimizer)
    return checkpoint.average_checkpoints(
        self._config["model_dir"],
        output_dir,
        trackables,
        max_count=max_count)

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
    infer_config = self._config["infer"]
    dataset = self._model.examples_inputter.make_inference_dataset(
        features_file,
        infer_config["batch_size"],
        bucket_width=infer_config["bucket_width"],
        num_threads=infer_config.get("num_threads", 1),
        prefetch_buffer_size=infer_config.get("prefetch_buffer_size"))

    @tf.function(input_signature=(dataset_util.input_signature_from_dataset(dataset),))
    def _infer(source):
      _, predictions = self._model(source)
      if "index" in source:
        predictions["index"] = source["index"]
      return predictions

    self._checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)

    if predictions_file:
      stream = io.open(predictions_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    ordered_writer = None
    write_fn = lambda prediction: (
        self._model.print_prediction(prediction, params=infer_config, stream=stream))

    total_time = 0
    total_tokens = 0
    total_examples = 0

    for source in dataset:
      start_time = time.time()
      predictions = _infer(source)
      end_time = time.time()
      predictions = {k:v.numpy() for k, v in six.iteritems(predictions)}
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

    if log_time:
      tf.get_logger().info("Total prediction time (s): %f", total_time)
      tf.get_logger().info(
          "Average prediction time (s): %f", total_time / total_examples)
      if total_tokens > 0:
        tf.get_logger().info("Tokens per second: %f", total_tokens / total_time)
    if predictions_file:
      stream.close()

  def export(self, checkpoint_path=None, export_dir_base=None):
    """Exports a model.

    Args:
      checkpoint_path: The checkpoint path to export. If ``None``, the latest is used.
      export_dir_base: The base directory in which a timestamped subdirectory
        containing the exported model will be created. Defaults to
        ``$MODEL_DIR/export/manual``.

    Returns:
      The string path to the exported directory.
    """
    raise NotImplementedError()

  def score(self, features_file, predictions_file, checkpoint_path=None, output_file=None):
    """Scores existing predictions.

    Args:
      features_file: The input file.
      predictions_file: The predictions file to score.
      checkpoint_path: Path of a specific checkpoint to use. If ``None``,
        the latest is used.
      output_file: The file where the scores are saved. Otherwise, they will be
        printed on the standard output.

    Raises:
      ValueError: if the model is not a sequence to sequence model or a
        language model.
      ValueError: if no checkpoint are found.
      ValueError: if :obj:`predictions_file` is not given.
    """
    if not isinstance(self._model, (models.LanguageModel, models.SequenceToSequence)):
      raise ValueError("scoring only works for sequence to sequence or language models")
    if isinstance(self._model, models.SequenceToSequence) and not predictions_file:
      raise ValueError("predictions_file is required when scoring with a "
                       "sequence to sequence model")

    self._checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
    score_config = self._config["score"]
    dataset = self._model.examples_inputter.make_evaluation_dataset(
        features_file,
        predictions_file,
        score_config["batch_size"],
        num_threads=score_config.get("num_threads"),
        prefetch_buffer_size=score_config.get("prefetch_buffer_size"))

    @tf.function(input_signature=dataset_util.input_signature_from_dataset(dataset))
    def _score(features, labels):
      outputs, _ = self._model(features, labels=labels, mode=tf.estimator.ModeKeys.EVAL)
      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels["ids_out"], outputs["logits"])
      weights = tf.sequence_mask(labels["length"], dtype=cross_entropy.dtype)
      masked_cross_entropy = cross_entropy * weights
      scores = tf.reduce_sum(masked_cross_entropy, axis=1)
      results = {
          "cross_entropy": cross_entropy,
          "score": scores,
          "tokens": labels["tokens"],
          "length": labels["length"] - 1  # -1 for the special token.
      }
      if "attention" in outputs:
        results["attention"] = outputs["attention"]
      return results

    if output_file:
      stream = io.open(output_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    output_tokenizer = (
        self._model.labels_inputter.tokenizer if not self._model.unsupervised
        else self._model.features_inputter.tokenizer)

    for source, target in dataset:
      results = _score(source, target)
      results = {k:v.numpy() for k, v in six.iteritems(results)}
      for batch in misc.extract_batches(results):
        tokens = batch["tokens"][:batch["length"]]
        sentence = output_tokenizer.detokenize(tokens)
        token_level_scores = None
        attention = None
        if score_config.get("with_token_level"):
          token_level_scores = batch["cross_entropy"][:batch["length"]]
        if "attention" in batch:
          attention = batch["attention"][:batch["length"]]
        alignment_type = score_config.get("with_alignments")
        sentence = misc.format_translation_output(
            sentence,
            score=batch["score"],
            token_level_scores=token_level_scores,
            attention=attention,
            alignment_type=alignment_type)
        misc.print_bytes(tf.compat.as_bytes(sentence), stream=stream)

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
                          sample_iterations=5,
                          num_devices=1,
                          gpu_memory_fraction=0.8):
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
    gpu_memory_fraction: Fraction of the GPU memory to use.

  Returns:
    The autotuned batch size.
  """
  config = copy.deepcopy(config)
  config["train"]["save_checkpoints_steps"] = None
  config["train"]["average_last_checkpoints"] = 0
  config["train"]["train_steps"] = sample_iterations
  config_path = os.path.join(config["model_dir"], "batch_size_autotuner.yml")

  # Define the TensorFlow session config, if needed.
  session_config_path = None
  if gpu_memory_fraction < 1:
    session_config = tf.compat.v1.ConfigProto(
        gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
    session_config_path = os.path.join(config["model_dir"], "batch_size_autotuner.proto")
    with tf.io.gfile.GFile(session_config_path, mode="w") as session_config_file:
      session_config_file.write(text_format.MessageToString(session_config))

  args = [
      "python", "-m", "opennmt.bin.main", "train",
      "--config", config_path, "--num_gpus", str(num_devices)]
  if session_config_path is not None:
    args += ["--session_config", session_config_path]

  tf.compat.v1.logging.info(
      "Searching the largest batch size between %d and %d with a precision of %d...",
      min_batch_size, max_batch_size, min_range)

  while max_batch_size - min_batch_size > min_range:
    batch_size = (max_batch_size + min_batch_size) // 2

    # Update configuration with current batch size and adjusted gradients
    # accumulation.
    config["train"]["batch_size"] = batch_size
    if config["train"].get("effective_batch_size") is not None:
      config["params"]["gradients_accum"] = _count_batch_accum(
          batch_size, config["train"]["effective_batch_size"], num_replicas=num_devices)
    with tf.io.gfile.GFile(config_path, mode="wb") as config_file:
      yaml.dump(config, config_file)

    tf.compat.v1.logging.info("Trying training with batch size %d...", batch_size)
    with open(os.devnull, "w") as devnull:
      process = subprocess.Popen(args, stdout=devnull, stderr=devnull)
      exit_code = process.wait()

    if exit_code != 0:
      tf.compat.v1.logging.info("... failed.")
      max_batch_size = batch_size - 1
    else:
      tf.compat.v1.logging.info(
          "... succeeded, continue until the search range is smaller than %d.", min_range)
      min_batch_size = batch_size

  tf.compat.v1.logging.info("Batch size auto tuned to %d.", min_batch_size)

  # Cleanup temporary files.
  os.remove(config_path)
  if session_config_path is not None:
    os.remove(session_config_path)
  return min_batch_size
