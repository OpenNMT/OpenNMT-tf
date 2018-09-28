"""Main library entrypoint."""

import io
import os
import sys
import random

import numpy as np
import tensorflow as tf

from tensorflow.python.estimator.util import fn_args

from google.protobuf import text_format

from opennmt.utils import hooks, checkpoint
from opennmt.utils.evaluator import external_evaluation_fn
from opennmt.utils.misc import extract_batches, print_bytes


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
               session_config=None):
    """Initializes the runner parameters.

    Args:
      model: A :class:`opennmt.models.model.Model` instance to run.
      config: The run configuration.
      seed: The random seed to set.
      num_devices: The number of devices (GPUs) to use for training.
      gpu_allow_growth: Allow GPU memory to grow dynamically.
      session_config: ``tf.ConfigProto`` overrides.
    """
    self._model = model
    self._config = config
    self._num_devices = num_devices

    session_config_base = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        gpu_options=tf.GPUOptions(
            allow_growth=gpu_allow_growth))

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
    session_config = session_config_base
    run_config = tf.estimator.RunConfig(
        model_dir=self._config["model_dir"],
        session_config=session_config,
        tf_random_seed=seed)

    # Create a first session to enforce GPU options.
    # See https://github.com/OpenNMT/OpenNMT-tf/issues/80.
    _ = tf.Session(config=session_config)

    np.random.seed(seed)
    random.seed(seed)

    if "train" in self._config:
      if "save_summary_steps" in self._config["train"]:
        run_config = run_config.replace(
            save_summary_steps=self._config["train"]["save_summary_steps"],
            log_step_count_steps=self._config["train"]["save_summary_steps"])
      if "save_checkpoints_steps" in self._config["train"]:
        run_config = run_config.replace(
            save_checkpoints_secs=None,
            save_checkpoints_steps=self._config["train"]["save_checkpoints_steps"])
      if "keep_checkpoint_max" in self._config["train"]:
        run_config = run_config.replace(
            keep_checkpoint_max=self._config["train"]["keep_checkpoint_max"])

    self._estimator = tf.estimator.Estimator(
        self._model.model_fn(num_devices=self._num_devices),
        config=run_config,
        params=self._config["params"])

  def _build_train_spec(self):
    train_hooks = [
        hooks.LogParametersCountHook(),
        hooks.CountersHook(
            every_n_steps=self._estimator.config.save_summary_steps,
            output_dir=self._estimator.model_dir)]

    train_spec = tf.estimator.TrainSpec(
        input_fn=self._model.input_fn(
            tf.estimator.ModeKeys.TRAIN,
            self._config["train"]["batch_size"],
            self._config["data"],
            self._config["data"]["train_features_file"],
            labels_file=self._config["data"]["train_labels_file"],
            batch_type=self._config["train"].get("batch_type", "examples"),
            batch_multiplier=self._num_devices,
            bucket_width=self._config["train"].get("bucket_width", 5),
            single_pass=self._config["train"].get("single_pass", False),
            num_threads=self._config["train"].get("num_threads"),
            sample_buffer_size=self._config["train"].get("sample_buffer_size", 500000),
            prefetch_buffer_size=self._config["train"].get("prefetch_buffer_size"),
            maximum_features_length=self._config["train"].get("maximum_features_length"),
            maximum_labels_length=self._config["train"].get("maximum_labels_length")),
        max_steps=self._config["train"].get("train_steps"),
        hooks=train_hooks)
    return train_spec

  def _build_eval_spec(self):
    if "eval" not in self._config:
      self._config["eval"] = {}

    eval_hooks = []
    if (self._config["eval"].get("save_eval_predictions", False)
        or self._config["eval"].get("external_evaluators") is not None):
      save_path = os.path.join(self._estimator.model_dir, "eval")
      if not os.path.isdir(save_path):
        os.makedirs(save_path)
      eval_hooks.append(hooks.SaveEvaluationPredictionHook(
          self._model,
          os.path.join(save_path, "predictions.txt"),
          post_evaluation_fn=external_evaluation_fn(
              self._config["eval"].get("external_evaluators"),
              self._config["data"]["eval_labels_file"],
              output_dir=self._estimator.model_dir)))

    eval_spec = tf.estimator.EvalSpec(
        input_fn=self._model.input_fn(
            tf.estimator.ModeKeys.EVAL,
            self._config["eval"].get("batch_size", 32),
            self._config["data"],
            self._config["data"]["eval_features_file"],
            num_threads=self._config["eval"].get("num_threads"),
            prefetch_buffer_size=self._config["eval"].get("prefetch_buffer_size"),
            labels_file=self._config["data"]["eval_labels_file"]),
        steps=None,
        hooks=eval_hooks,
        exporters=_make_exporters(
            self._config["eval"].get("exporters", "last"),
            self._model.serving_input_fn(self._config["data"])),
        throttle_secs=self._config["eval"].get("eval_delay", 18000))
    return eval_spec

  def train_and_evaluate(self):
    """Runs the training and evaluation loop."""
    train_spec = self._build_train_spec()
    eval_spec = self._build_eval_spec()
    tf.estimator.train_and_evaluate(self._estimator, train_spec, eval_spec)
    self._maybe_average_checkpoints()

  def train(self):
    """Runs the training loop."""
    train_spec = self._build_train_spec()
    self._estimator.train(
        train_spec.input_fn, hooks=train_spec.hooks, max_steps=train_spec.max_steps)
    self._maybe_average_checkpoints()

  def evaluate(self, checkpoint_path=None):
    """Runs evaluation."""
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    eval_spec = self._build_eval_spec()
    self._estimator.evaluate(
        eval_spec.input_fn, hooks=eval_spec.hooks, checkpoint_path=checkpoint_path)

  def _maybe_average_checkpoints(self, avg_subdirectory="avg"):
    """Averages checkpoints if enabled in the training configuration and if the
    current training instance is the chief.

    Args:
      avg_subdirectory: The directory within the model directory that will
        contain the averaged checkpoint.

    Returns:
      The path to the directory containing the averaged checkpoint or ``None``
      if no checkpoints were averaged.
    """
    average_last_checkpoints = self._config["train"].get("average_last_checkpoints", 0)
    if average_last_checkpoints > 0 and self._estimator.config.is_chief:
      return self.average_checkpoints(
          os.path.join(self._estimator.model_dir, avg_subdirectory),
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
        self._estimator.model_dir,
        output_dir,
        max_count=max_count,
        session_config=self._estimator.config.session_config)

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
    if "infer" not in self._config:
      self._config["infer"] = {}
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    batch_size = self._config["infer"].get("batch_size", 1)
    input_fn = self._model.input_fn(
        tf.estimator.ModeKeys.PREDICT,
        batch_size,
        self._config["data"],
        features_file,
        num_threads=self._config["infer"].get("num_threads"),
        prefetch_buffer_size=self._config["infer"].get("prefetch_buffer_size"))

    if predictions_file:
      stream = io.open(predictions_file, encoding="utf-8", mode="w")
    else:
      stream = sys.stdout

    infer_hooks = []
    if log_time:
      infer_hooks.append(hooks.LogPredictionTimeHook())

    for prediction in self._estimator.predict(
        input_fn=input_fn,
        checkpoint_path=checkpoint_path,
        hooks=infer_hooks):
      self._model.print_prediction(prediction, params=self._config["infer"], stream=stream)

    if predictions_file:
      stream.close()

  def export(self, checkpoint_path=None):
    """Exports a model.

    Args:
      checkpoint_path: The checkpoint path to export. If ``None``, the latest is used.

    Returns:
      The string path to the exported directory.
    """
    if checkpoint_path is not None and os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)

    export_dir = os.path.join(self._estimator.model_dir, "export")
    if not os.path.isdir(export_dir):
      os.makedirs(export_dir)

    kwargs = {}
    if "strip_default_attrs" in fn_args(self._estimator.export_savedmodel):
      # Set strip_default_attrs to True for TensorFlow 1.6+ to stay consistent
      # with the behavior of tf.estimator.Exporter.
      kwargs["strip_default_attrs"] = True

    return self._estimator.export_savedmodel(
        os.path.join(export_dir, "manual"),
        self._model.serving_input_fn(self._config["data"]),
        checkpoint_path=checkpoint_path,
        **kwargs)

  def score(self, features_file, predictions_file, checkpoint_path=None):
    """Scores existing predictions.

    Args:
      features_file: The input file.
      predictions_file: The predictions file to score.
      checkpoint_path: Path of a specific checkpoint to use. If ``None``,
        the latest is used.

    Raises:
      ValueError: if no checkpoint are found or if the model is not a sequence to
        sequence model.
    """
    if not hasattr(self._model, "target_inputter"):
      raise ValueError("scoring only works for sequence to sequence models")

    if checkpoint_path is None:
      checkpoint_path = tf.train.latest_checkpoint(self._estimator.model_dir)
    elif os.path.isdir(checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    if checkpoint_path is None:
      raise ValueError("could not find a trained model in %s" % self._estimator.model_dir)

    if "score" not in self._config:
      self._config["score"] = {}
    batch_size = self._config["score"].get("batch_size", 64)
    input_fn = self._model.input_fn(
        tf.estimator.ModeKeys.EVAL,
        batch_size,
        self._config["data"],
        features_file,
        labels_file=predictions_file,
        num_threads=self._config["score"].get("num_threads"),
        prefetch_buffer_size=self._config["score"].get("prefetch_buffer_size"))

    with tf.Graph().as_default() as g:
      tf.train.create_global_step(g)
      features, labels = input_fn()
      with tf.variable_scope(self._model.name):
        logits, _ = self._model(
            features,
            labels,
            self._estimator.params,
            tf.estimator.ModeKeys.EVAL)

      cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=logits, labels=labels["ids_out"])
      weights = tf.sequence_mask(labels["length"], dtype=cross_entropy.dtype)
      masked_cross_entropy = cross_entropy * weights
      scores = (tf.reduce_sum(masked_cross_entropy, axis=1) /
                tf.cast(labels["length"], cross_entropy.dtype))
      results = {
          "score": scores,
          "tokens": labels["tokens"],
          "length": labels["length"] - 1  # For -1, see sequence_to_sequence.shift_target_sequence.
      }

      with tf.train.MonitoredSession(
          session_creator=tf.train.ChiefSessionCreator(
              checkpoint_filename_with_path=checkpoint_path,
              config=self._estimator.config.session_config)) as sess:
        while not sess.should_stop():
          for batch in extract_batches(sess.run(results)):
            tokens = batch["tokens"][:batch["length"]]
            sentence = self._model.target_inputter.tokenizer.detokenize(tokens)
            fmt = "%f ||| %s" % (batch["score"], sentence)
            print_bytes(tf.compat.as_bytes(fmt))


def _make_exporters(exporters_type, serving_input_fn):
  if exporters_type is None:
    return None
  if not isinstance(exporters_type, list):
    exporters_type = [exporters_type]
  exporters = []
  for exporter_type in exporters_type:
    exporter_type = exporter_type.lower()
    if exporter_type == "last":
      exporters.append(tf.estimator.LatestExporter("latest", serving_input_fn))
    elif exporter_type == "final":
      exporters.append(tf.estimator.FinalExporter("final", serving_input_fn))
    elif exporter_type == "best":
      if not hasattr(tf.estimator, "BestExporter"):
        raise ValueError("BestExporter is only available starting from TensorFlow 1.9")
      exporters.append(tf.estimator.BestExporter(
          name="best", serving_input_receiver_fn=serving_input_fn))
    else:
      raise ValueError("invalid exporter type: %s" % exporter_type)
  if len(exporters) == 1:
    return exporters[0]
  return exporters
