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

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from google.protobuf import text_format

from opennmt import estimator as estimator_util
from opennmt import models
from opennmt.utils import hooks, checkpoint, misc
from opennmt.utils import evaluator
from opennmt.utils.misc import format_translation_output, OrderRestorer
from opennmt.utils.parallel import get_devices


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
  """Class for managing training, inference, and export. It is mostly a
  wrapper around ``tf.estimator.Estimator``.
  """

  def __init__(self,
               model,
               config,
               seed=None,
               num_devices=1,
               gpu_allow_growth=False,
               session_config=None,
               auto_config=False,
               hvd=None):
    """Initializes the runner parameters.

    Args:
      model: A :class:`opennmt.models.model.Model` instance to run.
      config: The run configuration.
      seed: The random seed to set.
      num_devices: The number of devices (GPUs) to use for training.
      gpu_allow_growth: Allow GPU memory to grow dynamically.
      session_config: ``tf.ConfigProto`` overrides.
      auto_config: If ``True``, use automatic configuration values defined by
        :obj:`model`.
      hvd: Optional Horovod object.

    Raises:
      NotImplementedError: If :obj:`auto_config` is ``True`` but :obj:`model`
        does not define any automatic configuration values.
    """
    self._model = model
    self._num_devices = num_devices
    self._num_replicas = hvd.size() if hvd is not None else num_devices
    self._seed = seed
    self._hvd = hvd

    # Configuration priority: user config > auto config > default config.
    self._config = copy.deepcopy(_CONFIG_FALLBACK)
    if auto_config:
      model_config = self._model.auto_config(num_devices=self._num_replicas)
      if not model_config:
        raise NotImplementedError("This model does not define any automatic configuration values")
      misc.merge_dict(self._config, model_config)
    misc.merge_dict(self._config, config)
    self._model.initialize(self._config["data"])
    tf.logging.info(
        "Using parameters:\n%s", yaml.dump(self._config, indent=2, default_flow_style=False))

    session_config_base = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=gpu_allow_growth))
    if self._hvd is not None:
      session_config_base.gpu_options.visible_device_list = str(self._hvd.local_rank())

    # Disable layout optimizer for better conv1d performance, see:
    # https://github.com/tensorflow/tensorflow/issues/20309
    # This field does not exist in TensorFlow 1.4, so guard against the
    # exception.
    try:
      rewrite_options = text_format.Parse("""
          graph_options {
            rewrite_options {
              layout_optimizer: OFF
            }
          }
          """, tf.ConfigProto())
      session_config_base.MergeFrom(rewrite_options)
    except text_format.ParseError:
      pass

    if session_config is not None:
      session_config_base.MergeFrom(session_config)
    self._session_config = session_config_base

    np.random.seed(seed)
    random.seed(seed)

  def _make_estimator(self):
    params = self._config["params"]
    train_config = self._config["train"]
    summary_steps = train_config["save_summary_steps"]

    run_config = tf.estimator.RunConfig(
        model_dir=self._config["model_dir"],
        tf_random_seed=self._seed,
        save_summary_steps=summary_steps,
        session_config=self._session_config,
        log_step_count_steps=params.get("gradients_accum", 1) * summary_steps)
    if "save_checkpoints_steps" in train_config or "save_checkpoints_secs" in train_config:
      run_config = run_config.replace(
          save_checkpoints_secs=train_config.get("save_checkpoints_secs"),
          save_checkpoints_steps=train_config.get("save_checkpoints_steps"))
    if not self.is_chief():
      run_config = run_config.replace(
          save_checkpoints_secs=None,
          save_checkpoints_steps=None)
    if "keep_checkpoint_max" in train_config:
      run_config = run_config.replace(
          keep_checkpoint_max=train_config["keep_checkpoint_max"])

    devices = get_devices(num_devices=self._num_devices, session_config=self._session_config)
    return tf.estimator.Estimator(
        estimator_util.make_model_fn(
            self._model,
            eval_prediction_hooks_fn=self._make_eval_prediction_hooks_fn(),
            devices=devices,
            hvd=self._hvd),
        config=run_config,
        params=params)

  def is_chief(self):
    """Returns ``True`` if this runner is the master runner."""
    if self._hvd is not None:
      return self._hvd.rank() == 0
    cluster_spec = os.getenv("TF_CONFIG")
    if cluster_spec is None:
      return True
    cluster_spec = json.loads(cluster_spec)
    return cluster_spec["task"]["type"] == "chief"

  def _make_eval_prediction_hooks_fn(self):
    if (not self._config["eval"].get("save_eval_predictions", False)
        and self._config["eval"].get("external_evaluators") is None):
      return None
    if self._model.unsupervised:
      raise RuntimeError("This model does not support saving evaluation predictions")
    save_path = os.path.join(self._config["model_dir"], "eval")
    if not tf.gfile.Exists(save_path):
      tf.gfile.MakeDirs(save_path)
    scorers = evaluator.make_scorers(self._config["eval"].get("external_evaluators"))
    external_evaluator = evaluator.ExternalEvaluator(
        labels_file=self._config["data"]["eval_labels_file"],
        output_dir=save_path,
        scorers=scorers)
    return lambda predictions: [
        hooks.SaveEvaluationPredictionHook(
            self._model,
            os.path.join(save_path, "predictions.txt"),
            post_evaluation_fn=external_evaluator,
            predictions=predictions)]

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
      tf.logging.info("Accumulate gradients of %d iterations to reach effective batch size of %d",
                      self._config["params"]["gradients_accum"],
                      train_config["effective_batch_size"])

  def _build_train_spec(self, checkpoint_path):
    train_hooks = [
        hooks.LogParametersCountHook()]

    if checkpoint_path is not None:
      train_hooks.append(hooks.LoadWeightsFromCheckpointHook(checkpoint_path))
    if self._hvd is not None:
      train_hooks.append(self._hvd.BroadcastGlobalVariablesHook(0))

    train_steps = self._config["train"].get("train_steps")
    if train_steps is not None and self._hvd is not None:
      train_steps //= self._hvd.size()
    train_spec = tf.estimator.TrainSpec(
        input_fn=estimator_util.make_input_fn(
            self._model,
            tf.estimator.ModeKeys.TRAIN,
            self._config["train"]["batch_size"],
            features_file=self._config["data"]["train_features_file"],
            labels_file=self._config["data"].get("train_labels_file"),
            batch_type=self._config["train"]["batch_type"],
            batch_multiplier=self._num_devices,
            bucket_width=self._config["train"]["bucket_width"],
            maximum_features_length=self._config["train"].get("maximum_features_length"),
            maximum_labels_length=self._config["train"].get("maximum_labels_length"),
            shuffle_buffer_size=self._config["train"]["sample_buffer_size"],
            single_pass=self._config["train"].get("single_pass", False),
            num_shards=self._hvd.size() if self._hvd is not None else 1,
            shard_index=self._hvd.rank() if self._hvd is not None else 0,
            num_threads=self._config["train"].get("num_threads"),
            prefetch_buffer_size=self._config["train"].get("prefetch_buffer_size"),
            return_dataset=False),
        max_steps=train_steps,
        hooks=train_hooks)
    return train_spec

  def _build_eval_spec(self):
    eval_spec = tf.estimator.EvalSpec(
        input_fn=estimator_util.make_input_fn(
            self._model,
            tf.estimator.ModeKeys.EVAL,
            self._config["eval"]["batch_size"],
            features_file=self._config["data"]["eval_features_file"],
            labels_file=self._config["data"].get("eval_labels_file"),
            num_threads=self._config["eval"].get("num_threads"),
            prefetch_buffer_size=self._config["eval"].get("prefetch_buffer_size"),
            return_dataset=False),
        steps=None,
        exporters=_make_exporters(
            self._config["eval"]["exporters"],
            estimator_util.make_serving_input_fn(self._model, metadata=self._config["data"]),
            assets_extra=self._get_model_assets()),
        throttle_secs=self._config["eval"]["eval_delay"])
    return eval_spec

  def _get_model_assets(self):
    generated_assets_path = os.path.join(self._config["model_dir"], "assets")
    if not tf.gfile.Exists(generated_assets_path):
      tf.gfile.MakeDirs(generated_assets_path)
    return self._model.get_assets(asset_dir=generated_assets_path)

  def train_and_evaluate(self, checkpoint_path=None):
    """Runs the training and evaluation loop.

    Args:
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      A tuple with a dict of evaluation metrics and the export result or
      ``None`` in TensorFlow 1.8 and older.
    """
    if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    self._finalize_training_parameters()
    train_spec = self._build_train_spec(checkpoint_path)
    eval_spec = self._build_eval_spec()
    estimator = self._make_estimator()
    result = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    self._maybe_average_checkpoints(estimator)
    return result

  def train(self, checkpoint_path=None):
    """Runs the training loop.

    Args:
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      The path to the final model directory.
    """
    if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    self._finalize_training_parameters()
    train_spec = self._build_train_spec(checkpoint_path)
    estimator = self._make_estimator()
    estimator.train(
        train_spec.input_fn, hooks=train_spec.hooks, max_steps=train_spec.max_steps)
    output_dir = self._maybe_average_checkpoints(estimator)
    if output_dir is None:
      output_dir = estimator.model_dir
    return output_dir

  def evaluate(self, checkpoint_path=None):
    """Runs evaluation.

    Args:
      checkpoint_path: The checkpoint path to load the model weights from it.

    Returns:
      A dict of evaluation metrics.
    """
    if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    eval_spec = self._build_eval_spec()
    estimator = self._make_estimator()
    return estimator.evaluate(
        eval_spec.input_fn, hooks=eval_spec.hooks, checkpoint_path=checkpoint_path)

  def _maybe_average_checkpoints(self, estimator, avg_subdirectory="avg"):
    """Averages checkpoints if enabled in the training configuration and if the
    current training instance is the chief.

    Args:
      estimator: The ``tf.estimator.Estimator`` instance used for the training.
      avg_subdirectory: The directory within the model directory that will
        contain the averaged checkpoint.

    Returns:
      The path to the directory containing the averaged checkpoint or ``None``
      if no checkpoints were averaged.
    """
    average_last_checkpoints = self._config["train"].get("average_last_checkpoints", 0)
    if average_last_checkpoints > 0 and self.is_chief():
      return self.average_checkpoints(
          os.path.join(estimator.model_dir, avg_subdirectory),
          max_count=average_last_checkpoints)
    return None

  def average_checkpoints(self, output_dir, max_count=8):
    """Averages checkpoints.

    Args:
      output_dir: The directory that will contain the averaged checkpoint.
      max_count: The maximum number of checkpoints to average.

    Returns:
      The path to the directory containing the averaged checkpoint.
    """
    return checkpoint.average_checkpoints(
        self._config["model_dir"],
        output_dir,
        max_count=max_count,
        session_config=self._session_config)

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
    if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    input_fn = estimator_util.make_input_fn(
        self._model,
        tf.estimator.ModeKeys.PREDICT,
        self._config["infer"]["batch_size"],
        features_file=features_file,
        bucket_width=self._config["infer"]["bucket_width"],
        num_threads=self._config["infer"].get("num_threads"),
        prefetch_buffer_size=self._config["infer"].get("prefetch_buffer_size"),
        return_dataset=False)

    if predictions_file:
      stream = io.open(predictions_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    infer_hooks = []
    if log_time:
      infer_hooks.append(hooks.LogPredictionTimeHook())

    ordered_writer = None
    write_fn = lambda prediction: (
        self._model.print_prediction(prediction, params=self._config["infer"], stream=stream))

    estimator = self._make_estimator()
    for prediction in estimator.predict(
        input_fn=input_fn,
        checkpoint_path=checkpoint_path,
        hooks=infer_hooks):
      # If the index is part of the prediction, they may be out of order.
      if "index" in prediction:
        if ordered_writer is None:
          ordered_writer = OrderRestorer(
              index_fn=lambda prediction: prediction["index"], callback_fn=write_fn)
        ordered_writer.push(prediction)
      else:
        write_fn(prediction)

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
    estimator = self._make_estimator()
    if checkpoint_path is not None and tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if export_dir_base is None:
      export_dir_base = os.path.join(estimator.model_dir, "export", "manual")

    kwargs = {}
    if hasattr(estimator, "export_saved_model"):
      export_fn = estimator.export_saved_model
    else:
      export_fn = estimator.export_savedmodel
      if "strip_default_attrs" in fn_args(estimator.export_savedmodel):
        # Set strip_default_attrs to True for TensorFlow 1.6+ to stay consistent
        # with the behavior of tf.estimator.Exporter.
        kwargs["strip_default_attrs"] = True

    return export_fn(
        export_dir_base,
        estimator_util.make_serving_input_fn(self._model, metadata=self._config["data"]),
        assets_extra=self._get_model_assets(),
        checkpoint_path=checkpoint_path,
        **kwargs)

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
      ValueError: if no checkpoint are found or if the model is not a sequence to
        sequence model.
    """
    if not isinstance(self._model, (models.LanguageModel, models.SequenceToSequence)):
      raise ValueError("scoring only works for sequence to sequence or language models")

    if checkpoint_path is None:
      checkpoint_path = tf.train.latest_checkpoint(self._config["model_dir"])
    elif tf.gfile.IsDirectory(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_path is None:
      raise ValueError("could not find a trained model in %s" % self._config["model_dir"])

    model = copy.deepcopy(self._model)
    with tf.Graph().as_default():
      dataset = model.examples_inputter.make_evaluation_dataset(
          features_file,
          predictions_file,
          self._config["score"]["batch_size"],
          num_threads=self._config["score"].get("num_threads"),
          prefetch_buffer_size=self._config["score"].get("prefetch_buffer_size"))
      iterator = dataset.make_initializable_iterator()
      features, labels = iterator.get_next()
      labels["alignment"] = None  # Add alignment key to force the model to return attention.
      outputs, _ = model(
          features,
          labels,
          self._config["params"],
          tf.estimator.ModeKeys.EVAL)

      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=outputs["logits"], labels=labels["ids_out"])
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

      if output_file:
        stream = io.open(output_file, encoding="utf-8", mode="w")
      else:
        stream = sys.stdout

      output_tokenizer = (
          self._model.labels_inputter.tokenizer if not self._model.unsupervised
          else self._model.features_inputter.tokenizer)
      with tf.train.MonitoredSession(
          session_creator=tf.train.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              config=self._session_config)) as sess:
        sess.run(iterator.initializer)
        while not sess.should_stop():
          for batch in misc.extract_batches(sess.run(results)):
            tokens = batch["tokens"][:batch["length"]]
            sentence = output_tokenizer.detokenize(tokens)
            token_level_scores = None
            attention = None
            if self._config["score"].get("with_token_level"):
              token_level_scores = batch["cross_entropy"][:batch["length"]]
            if "attention" in batch:
              attention = batch["attention"][:batch["length"]]
            alignment_type = self._config["score"].get("with_alignments")
            sentence = format_translation_output(
                sentence,
                score=batch["score"],
                token_level_scores=token_level_scores,
                attention=attention,
                alignment_type=alignment_type)
            misc.print_bytes(tf.compat.as_bytes(sentence), stream=stream)

      if output_file:
        stream.close()

def _make_exporters(exporters_type, serving_input_fn, assets_extra=None):
  if exporters_type is None:
    return None
  if not isinstance(exporters_type, list):
    exporters_type = [exporters_type]
  exporters = []
  for exporter_type in exporters_type:
    exporter_type = exporter_type.lower()
    if exporter_type == "last":
      exporters.append(tf.estimator.LatestExporter(
          "latest", serving_input_fn, assets_extra=assets_extra))
    elif exporter_type == "final":
      exporters.append(tf.estimator.FinalExporter(
          "final", serving_input_fn, assets_extra=assets_extra))
    elif exporter_type == "best":
      if not hasattr(tf.estimator, "BestExporter"):
        raise ValueError("BestExporter is only available starting from TensorFlow 1.9")
      exporters.append(tf.estimator.BestExporter(
          name="best", serving_input_receiver_fn=serving_input_fn, assets_extra=assets_extra))
    else:
      raise ValueError("invalid exporter type: %s" % exporter_type)
  if len(exporters) == 1:
    return exporters[0]
  return exporters

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
    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
    session_config_path = os.path.join(config["model_dir"], "batch_size_autotuner.proto")
    with tf.gfile.Open(session_config_path, mode="w") as session_config_file:
      session_config_file.write(text_format.MessageToString(session_config))

  args = [
      "python", "-m", "opennmt.bin.main", "train",
      "--config", config_path, "--num_gpus", str(num_devices)]
  if session_config_path is not None:
    args += ["--session_config", session_config_path]

  tf.logging.info("Searching the largest batch size between %d and %d with a precision of %d...",
                  min_batch_size, max_batch_size, min_range)

  while max_batch_size - min_batch_size > min_range:
    batch_size = (max_batch_size + min_batch_size) // 2

    # Update configuration with current batch size and adjusted gradients
    # accumulation.
    config["train"]["batch_size"] = batch_size
    if config["train"].get("effective_batch_size") is not None:
      config["params"]["gradients_accum"] = _count_batch_accum(
          batch_size, config["train"]["effective_batch_size"], num_replicas=num_devices)
    with tf.gfile.Open(config_path, mode="wb") as config_file:
      yaml.dump(config, config_file)

    tf.logging.info("Trying training with batch size %d...", batch_size)
    with open(os.devnull, "w") as devnull:
      process = subprocess.Popen(args, stdout=devnull, stderr=devnull)
      exit_code = process.wait()

    if exit_code != 0:
      tf.logging.info("... failed.")
      max_batch_size = batch_size - 1
    else:
      tf.logging.info(
          "... succeeded, continue until the search range is smaller than %d.", min_range)
      min_batch_size = batch_size

  tf.logging.info("Batch size auto tuned to %d.", min_batch_size)

  # Cleanup temporary files.
  os.remove(config_path)
  if session_config_path is not None:
    os.remove(session_config_path)
  return min_batch_size
