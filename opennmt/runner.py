"""Main library entrypoint."""

import copy
import os
import sys
import random
import math
import shutil
import subprocess
import tempfile
import yaml

import numpy as np
import tensorflow as tf

from opennmt import evaluation
from opennmt import inference
from opennmt import models
from opennmt import training as training_util
from opennmt.config import MODEL_DESCRIPTION_FILENAME
from opennmt.utils import checkpoint as checkpoint_util
from opennmt.utils import misc
from opennmt.version import __version__


# These options require a value but we can fallback to a default one.
_CONFIG_FALLBACK = {
    "params": {},
    "train": {
        "batch_type": "examples",
        "length_bucket_width": 1,
        "sample_buffer_size": 500000,
        "save_summary_steps": 100,
    },
    "eval": {
        "length_bucket_width": None,
        "batch_type": "examples",
        "batch_size": 32,
    },
    "infer": {
        "length_bucket_width": None,
        "batch_type": "examples",
        "batch_size": 16,
    },
    "score": {
        "length_bucket_width": None,
        "batch_type": "examples",
        "batch_size": 64,
    },
}


class Runner(object):
    """Class for running and exporting models."""

    def __init__(
        self, model, config, auto_config=False, mixed_precision=False, seed=None
    ):
        """Initializes the runner parameters.

        Args:
          model: A :class:`opennmt.models.Model` instance to run or a callable that
            returns such instance.
          config: The run configuration.
          auto_config: If ``True``, use automatic configuration values defined by
            :obj:`model`.
          mixed_precision: Enable mixed precision.
          seed: The random seed to set.

        Raises:
          TypeError: if :obj:`model` is not a :class:`opennmt.models.Model` instance
            or a callable.
        """
        if isinstance(model, models.Model):
            self._model = model
            self._model_fn = lambda: misc.clone_layer(model)
        elif callable(model):
            self._model = model()
            self._model_fn = model
        else:
            raise TypeError(
                "model should be a opennmt.models.Model instance or a callable"
            )
        tf.get_logger().info("Using OpenNMT-tf version %s", __version__)
        tf.get_logger().info("Using model:\n%s", self._model)
        self._optimizer = None
        self._config = copy.deepcopy(config)
        self._auto_config = auto_config
        self._mixed_precision = mixed_precision
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)

    @property
    def model(self):
        """The :class:`opennmt.models.Model` executed by this runner."""
        return self._model

    @property
    def model_dir(self):
        """The active model directory."""
        return self._config["model_dir"]

    def _finalize_config(self, training=False, num_replicas=1, num_devices=1):
        # Configuration priority: user config > auto config > default config.
        config = copy.deepcopy(_CONFIG_FALLBACK)
        if self._auto_config:
            model_config = self._model.auto_config(num_replicas=num_replicas)
            if not model_config:
                raise NotImplementedError(
                    "This model does not define any automatic configuration values"
                )
            misc.merge_dict(config, model_config)
        misc.merge_dict(config, self._config)

        config["params"].setdefault("num_hypotheses", config["infer"].get("n_best", 1))
        config["params"].setdefault(
            "average_loss_in_time", config["train"]["batch_type"] == "tokens"
        )

        if training:
            train_config = config["train"]
            batch_size = train_config.get("batch_size")

            # Auto tune batch size.
            if batch_size is None or batch_size == 0:
                if train_config["batch_type"] == "examples":
                    min_batch_size = 1
                    max_batch_size = 512
                    min_range = 16
                else:
                    min_batch_size = 256
                    max_batch_size = 16384
                    min_range = 256

                if train_config.get("effective_batch_size") is not None:
                    max_batch_size = min(
                        max_batch_size, train_config["effective_batch_size"]
                    )
                train_config["batch_size"] = _auto_tune_batch_size(
                    config,
                    min_batch_size=min_batch_size,
                    max_batch_size=max_batch_size,
                    min_range=min_range,
                    num_devices=num_devices,
                    scaling_factor=train_config.get("batch_size_autotune_scale", 0.7),
                    mixed_precision=self._mixed_precision,
                )

        tf.get_logger().info(
            "Using parameters:\n%s",
            yaml.dump(config, indent=2, default_flow_style=False),
        )
        return config

    def _init_model(self, config):
        model = self._model_fn()
        model.initialize(config["data"], params=config["params"])
        return model

    def train(
        self,
        num_devices=1,
        with_eval=False,
        checkpoint_path=None,
        hvd=None,
        return_summary=False,
        fallback_to_cpu=True,
    ):
        """Runs the training loop.

        Args:
          num_devices: Number of devices to use for training.
          with_eval: Enable evaluation during training.
          checkpoint_path: The checkpoint path to load the model weights from.
          hvd: Optional Horovod module.
          return_summary: Return a summary of the training from this function.
          fallback_to_cpu: If no GPU is detected, allow the training to run on CPU.

        Returns:
          The path to the final model directory and, if :obj:`return_summary` is set,
          a dictionary with various training statistics.
        """
        if hvd is None:
            num_replicas = num_devices
            is_master = True
        else:
            if num_devices > 1:
                raise ValueError(
                    "num_devices (or num_gpus) should be set to 1 when using Horovod"
                )
            num_replicas = hvd.size()
            is_master = hvd.rank() == 0

        devices = misc.get_devices(count=num_devices, fallback_to_cpu=fallback_to_cpu)

        config = self._finalize_config(
            training=True, num_replicas=num_replicas, num_devices=num_devices
        )

        mixed_precision = self._mixed_precision and misc.enable_mixed_precision()
        model = self._init_model(config)
        optimizer = model.get_optimizer()

        data_config = config["data"]
        train_config = config["train"]
        eval_config = config["eval"]

        batch_type = train_config["batch_type"]
        batch_size_multiple = 8 if mixed_precision and batch_type == "tokens" else 1

        dataset_fn = (
            lambda input_context: model.examples_inputter.make_training_dataset(
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
                num_shards=input_context.num_input_pipelines,
                shard_index=input_context.input_pipeline_id,
                prefetch_buffer_size=train_config.get("prefetch_buffer_size"),
                cardinality_multiple=input_context.num_replicas_in_sync,
                weights=data_config.get("train_files_weights"),
                batch_autotune_mode=train_config.get("batch_autotune_mode"),
            )
        )

        checkpoint = None
        evaluator = None
        if is_master:
            checkpoint = checkpoint_util.Checkpoint.from_config(
                config, model, optimizer=optimizer
            )
            checkpoint.restore(
                checkpoint_path=checkpoint_path,
                weights_only=checkpoint_path is not None,
            )
            if with_eval:
                evaluator = evaluation.Evaluator.from_config(model, config)

        # Set gradients accumulation based on the requested effective batch size.
        if train_config.get("effective_batch_size") is not None:
            accum_steps = _count_batch_accum(
                train_config["batch_size"],
                train_config["effective_batch_size"],
                num_replicas=num_replicas,
            )
            tf.get_logger().info(
                "Accumulate gradients of %d iterations to reach effective batch size of %d",
                accum_steps,
                train_config["effective_batch_size"],
            )
        else:
            accum_steps = 1

        if hvd is not None:
            trainer = training_util.HorovodTrainer(
                model, optimizer, hvd, checkpoint=checkpoint
            )
        elif num_devices > 1:
            trainer = training_util.MirroredStrategyTrainer(
                model, optimizer, checkpoint=checkpoint, devices=devices
            )
        else:
            trainer = training_util.Trainer(model, optimizer, checkpoint=checkpoint)

        summary = trainer(
            dataset_fn,
            max_step=train_config.get("max_step"),
            accum_steps=accum_steps,
            report_steps=train_config.get("save_summary_steps", 100),
            save_steps=train_config.get("save_checkpoints_steps", 5000),
            evaluator=evaluator,
            eval_steps=eval_config.get("steps", 5000),
            moving_average_decay=train_config.get("moving_average_decay"),
        )

        average_last_checkpoints = train_config.get("average_last_checkpoints", 0)
        if checkpoint is None:
            output_dir = None
        elif average_last_checkpoints > 0:
            output_dir = self.average_checkpoints(
                os.path.join(checkpoint.model_dir, "avg"),
                max_count=average_last_checkpoints,
            )
        else:
            output_dir = checkpoint.model_dir

        if mixed_precision:
            misc.disable_mixed_precision()

        if return_summary:
            return output_dir, summary
        return output_dir

    def evaluate(self, features_file=None, labels_file=None, checkpoint_path=None):
        """Runs evaluation.

        Args:
          features_file: The input features file to evaluate. If not set, will load
            ``eval_features_file`` from the data configuration.
          labels_file: The output labels file to evaluate. If not set, will load
            ``eval_labels_file`` from the data configuration.
          checkpoint_path: The checkpoint path to load the model weights from.

        Returns:
          A dict of evaluation metrics.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint_path = checkpoint.restore(
            checkpoint_path=checkpoint_path, weights_only=True
        )
        step = checkpoint_util.get_step_from_checkpoint_prefix(checkpoint_path)
        evaluator = evaluation.Evaluator.from_config(
            model, config, features_file=features_file, labels_file=labels_file
        )
        return evaluator(step)

    def average_checkpoints(self, output_dir, max_count=8):
        """Averages checkpoints.

        Args:
          output_dir: The directory that will contain the averaged checkpoint.
          max_count: The maximum number of checkpoints to average.

        Returns:
          The path to the directory containing the averaged checkpoint.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        optimizer = model.get_optimizer()
        checkpoint = checkpoint_util.Checkpoint.from_config(
            config, model, optimizer=optimizer
        )
        checkpoint.restore()
        model.create_variables(optimizer=optimizer)
        trackables = dict(model=model, optimizer=optimizer)
        output_dir = checkpoint_util.average_checkpoints(
            checkpoint.model_dir, output_dir, trackables, max_count=max_count
        )
        _forward_model_description(self.model_dir, output_dir)
        self._config["model_dir"] = output_dir
        return output_dir

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
            raise ValueError(
                "Updating vocabularies is only supported for sequence to sequence models"
            )
        config = self._finalize_config()
        if src_vocab is None and tgt_vocab is None:
            return config["model_dir"]

        model = self._init_model(config)
        optimizer = model.get_optimizer()
        cur_checkpoint = checkpoint_util.Checkpoint.from_config(
            config, model, optimizer=optimizer
        )
        cur_checkpoint.restore()
        model.create_variables(optimizer=optimizer)
        source_dir = self.model_dir

        self._config["model_dir"] = output_dir
        if src_vocab is not None:
            self._config["data"]["source_vocabulary"] = src_vocab
        if tgt_vocab is not None:
            self._config["data"]["target_vocabulary"] = tgt_vocab
        new_config = self._finalize_config()
        new_model = self._init_model(new_config)
        new_optimizer = new_model.get_optimizer()
        new_checkpoint = checkpoint_util.Checkpoint.from_config(
            new_config, new_model, optimizer=new_optimizer
        )
        new_model.create_variables(optimizer=new_optimizer)

        model.transfer_weights(
            new_model, new_optimizer=new_optimizer, optimizer=optimizer
        )
        new_optimizer.iterations.assign(optimizer.iterations)
        new_checkpoint.save()
        _forward_model_description(source_dir, output_dir)
        return output_dir

    def infer(
        self, features_file, predictions_file=None, checkpoint_path=None, log_time=False
    ):
        """Runs inference.

        Args:
          features_file: The file(s) to infer from.
          predictions_file: If set, predictions are saved in this file, otherwise
            they are printed on the standard output.
          checkpoint_path: Path to a specific checkpoint to load. If ``None``,
            the latest is used.
          log_time: If ``True``, several time metrics will be printed in the logs at
            the end of the inference loop.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        infer_config = config["infer"]
        dataset = model.examples_inputter.make_inference_dataset(
            features_file,
            infer_config["batch_size"],
            batch_type=infer_config["batch_type"],
            length_bucket_width=infer_config["length_bucket_width"],
            prefetch_buffer_size=infer_config.get("prefetch_buffer_size"),
        )
        inference.predict_dataset(
            model,
            dataset,
            print_params=infer_config,
            predictions_file=predictions_file,
            log_time=log_time,
        )

    def export(self, export_dir, checkpoint_path=None, exporter=None):
        """Exports a model.

        Args:
          export_dir: The export directory.
          checkpoint_path: The checkpoint path to export. If ``None``, the latest is used.
          exporter: A :class:`opennmt.utils.Exporter` instance. Defaults to
            :class:`opennmt.utils.SavedModelExporter`.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        model.export(export_dir, exporter=exporter)

    def score(
        self, features_file, predictions_file, checkpoint_path=None, output_file=None
    ):
        """Scores existing predictions.

        Args:
          features_file: The input file.
          predictions_file: The predictions file to score.
          checkpoint_path: Path to specific checkpoint to load. If ``None``,
            the latest is used.
          output_file: The file where the scores are saved. Otherwise, they will be
            printed on the standard output.
        """
        config = self._finalize_config()
        model = self._init_model(config)
        checkpoint = checkpoint_util.Checkpoint.from_config(config, model)
        checkpoint.restore(checkpoint_path=checkpoint_path, weights_only=True)
        score_config = config["score"]
        dataset = model.examples_inputter.make_evaluation_dataset(
            features_file,
            predictions_file,
            score_config["batch_size"],
            batch_type=score_config["batch_type"],
            length_bucket_width=score_config["length_bucket_width"],
            prefetch_buffer_size=score_config.get("prefetch_buffer_size"),
        )
        inference.score_dataset(
            model, dataset, print_params=score_config, output_file=output_file
        )


def _forward_model_description(source, destination):
    source = os.path.join(source, MODEL_DESCRIPTION_FILENAME)
    if os.path.isfile(source):
        if not os.path.isdir(destination):
            os.makedirs(destination)
        destination = os.path.join(destination, MODEL_DESCRIPTION_FILENAME)
        shutil.copyfile(source, destination)


def _count_batch_accum(batch_size, target_batch_size, num_replicas=1):
    """Given the current batch size, the number of replicas, and the requested
    effective batch size, returns the number of gradients to accumulate.
    """
    return int(math.ceil(float(target_batch_size) / (batch_size * num_replicas)))


def _auto_tune_batch_size(
    config,
    min_batch_size,
    max_batch_size,
    min_range,
    sample_iterations=5,
    num_devices=1,
    scaling_factor=0.7,
    mixed_precision=False,
):
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
      mixed_precision: If ``True``, run the autotuning with mixed precision.

    Returns:
      The autotuned batch size.
    """
    tf.get_logger().info(
        "Searching the largest batch size between %d and %d with a precision of %d...",
        min_batch_size,
        max_batch_size,
        min_range,
    )

    model_description = os.path.join(config["model_dir"], MODEL_DESCRIPTION_FILENAME)
    absolute_min_batch_size = min_batch_size

    while max_batch_size - min_batch_size > min_range:
        batch_size = (max_batch_size + min_batch_size) // 2

        with tempfile.TemporaryDirectory() as tmpdir:
            run_config = copy.deepcopy(config)
            run_config["model_dir"] = tmpdir
            run_config["train"]["batch_autotune_mode"] = True
            run_config["train"]["batch_size"] = batch_size
            run_config["train"]["save_checkpoints_steps"] = None
            run_config["train"]["average_last_checkpoints"] = 0
            run_config["train"]["max_step"] = sample_iterations

            config_path = os.path.join(tmpdir, "batch_size_autotuner.yml")
            with tf.io.gfile.GFile(config_path, mode="w") as config_file:
                yaml.dump(run_config, config_file)

            args = [
                sys.executable or "python",
                "-m",
                "opennmt.bin.main",
                "--config",
                config_path,
                "--model",
                model_description,
            ]
            if mixed_precision:
                args.extend(["--mixed_precision"])
            args.extend(
                [
                    "train",
                    "--num_gpus",
                    str(num_devices),
                ]
            )

            tf.get_logger().info("Trying training with batch size %d...", batch_size)
            with open(os.devnull, "w") as devnull:
                process = subprocess.Popen(args, stdout=devnull, stderr=devnull)
                exit_code = process.wait()

            if exit_code != 0:
                tf.get_logger().info("... failed.")
                max_batch_size = batch_size - 1
            else:
                tf.get_logger().info(
                    "... succeeded, continue until the search range is smaller than %d.",
                    min_range,
                )
                min_batch_size = batch_size

    if min_batch_size == absolute_min_batch_size:
        raise RuntimeError(
            "Batch size autotuning failed: all training attempts exited with an error. "
            "Either there is not enough memory to train this model, or unexpected errors "
            "occured. Please try to set a fixed batch size in the training configuration."
        )

    batch_size = max(int(scaling_factor * min_batch_size), absolute_min_batch_size)
    tf.get_logger().info("Batch size auto tuned to %d.", batch_size)
    return batch_size
