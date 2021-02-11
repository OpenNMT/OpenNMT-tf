"""Evaluation related classes and functions."""

import collections
import os
import shutil

import tensorflow as tf

from opennmt.utils import exporters
from opennmt.utils import misc
from opennmt.utils import scorers as scorers_lib


_SUMMARIES_SCOPE = "metrics"


class EarlyStopping(
    collections.namedtuple("EarlyStopping", ("metric", "min_improvement", "steps"))
):
    """Conditions for early stopping."""


class Evaluator(object):
    """Model evaluator."""

    def __init__(
        self,
        model,
        features_file,
        labels_file,
        batch_size,
        batch_type="examples",
        length_bucket_width=None,
        scorers=None,
        save_predictions=False,
        early_stopping=None,
        model_dir=None,
        export_on_best=None,
        exporter=None,
        max_exports_to_keep=5,
    ):
        """Initializes the evaluator.

        Args:
          model: A :class:`opennmt.models.Model` to evaluate.
          features_file: Path to the evaluation features.
          labels_file: Path to the evaluation labels.
          batch_size: The evaluation batch size.
          batch_type: The batching strategy to use: can be "examples" or "tokens".
          length_bucket_width: The width of the length buckets to select batch
            candidates from (for efficiency). Set ``None`` to not constrain batch
            formation.
          scorers: A list of scorers, callables taking the path to the reference and
            the hypothesis and return one or more scores.
          save_predictions: Save evaluation predictions to a file. This is ``True``
            when :obj:`scorers` is set.
          early_stopping: An ``EarlyStopping`` instance.
          model_dir: The active model directory.
          export_on_best: Export a model when this evaluation metric has the
            best value so far.
          exporter: A :class:`opennmt.utils.Exporter` instance to export the model.
            Defaults to :class:`opennmt.utils.SavedModelExporter`.
          max_exports_to_keep: Maximum number of exports to keep. Older exports will
            be garbage collected. Set to ``None`` to keep all exports.

        Raises:
          ValueError: If :obj:`save_predictions` is set but the model is not compatible.
          ValueError: If :obj:`save_predictions` is set but :obj:`model_dir` is ``None``.
          ValueError: If :obj:`export_on_best` is set but :obj:`model_dir` is ``None``.
          ValueError: If the :obj:`early_stopping` configuration is invalid.
        """
        if model_dir is not None:
            export_dir = os.path.join(model_dir, "export")
            eval_dir = os.path.join(model_dir, "eval")
        else:
            if save_predictions:
                raise ValueError(
                    "Saving evaluation predictions requires model_dir to be set"
                )
            if export_on_best is not None:
                raise ValueError("Exporting models requires model_dir to be set")
            export_dir = None
            eval_dir = None

        if scorers is None:
            scorers = []
        if scorers:
            save_predictions = True
        if save_predictions:
            if model.unsupervised:
                raise ValueError(
                    "This model does not support saving evaluation predictions"
                )
            if not tf.io.gfile.exists(eval_dir):
                tf.io.gfile.makedirs(eval_dir)
        self._model = model
        self._labels_file = labels_file
        self._save_predictions = save_predictions
        self._scorers = scorers
        self._eval_dir = eval_dir
        self._metrics_history = []
        if eval_dir is not None:
            self._summary_writer = tf.summary.create_file_writer(eval_dir)
            summaries = misc.read_summaries(eval_dir)
            for step, values in summaries:
                metrics = misc.extract_prefixed_keys(values, _SUMMARIES_SCOPE + "/")
                self._metrics_history.append((step, metrics))
        else:
            self._summary_writer = tf.summary.create_noop_writer()
        dataset = model.examples_inputter.make_evaluation_dataset(
            features_file,
            labels_file,
            batch_size,
            batch_type=batch_type,
            length_bucket_width=length_bucket_width,
            num_threads=1,
            prefetch_buffer_size=1,
        )

        self._eval_fn = tf.function(
            model.evaluate, input_signature=dataset.element_spec
        )
        self._dataset = dataset

        self._metrics_name = {"loss", "perplexity"}
        for scorer in self._scorers:
            self._metrics_name.update(scorer.scores_name)
        model_metrics = self._model.get_metrics()
        if model_metrics:
            self._metrics_name.update(set(model_metrics.keys()))

        if early_stopping is not None:
            if early_stopping.metric not in self._metrics_name:
                raise ValueError(
                    "Invalid early stopping metric '%s', expected one in %s"
                    % (early_stopping.metric, str(self._metrics_name))
                )
            if early_stopping.steps <= 0:
                raise ValueError("Early stopping steps should greater than 0")
        self._early_stopping = early_stopping

        self._export_on_best = export_on_best
        self._exporter = exporter
        self._export_dir = export_dir
        self._max_exports_to_keep = max_exports_to_keep

    @classmethod
    def from_config(cls, model, config, features_file=None, labels_file=None):
        """Creates an evaluator from the configuration.

        Args:
          model: A :class:`opennmt.models.Model` to evaluate.
          config: The global user configuration.
          features_file: Optional input features file to evaluate. If not set, will
            load ``eval_features_file`` from the data configuration.
          labels_file: Optional output labels file to evaluate. If not set, will load
            ``eval_labels_file`` from the data configuration.

        Returns:
          A :class:`opennmt.evaluation.Evaluator` instance.

        Raises:
          ValueError: for supervised models, if one of :obj:`features_file` and
            :obj:`labels_file` is set but not the other.
          ValueError: for unsupervised models, if :obj:`labels_file` is set.
        """
        if model.unsupervised:
            if labels_file is not None:
                raise ValueError(
                    "labels_file can not be set when evaluating unsupervised models"
                )
        elif (features_file is None) != (labels_file is None):
            raise ValueError(
                "features_file and labels_file should be both set for evaluation"
            )
        eval_config = config["eval"]
        scorers = eval_config.get("scorers")
        if scorers is None:
            # Backward compatibility with previous field name.
            scorers = eval_config.get("external_evaluators")
        if scorers is not None:
            scorers = scorers_lib.make_scorers(scorers)
        early_stopping_config = eval_config.get("early_stopping")
        if early_stopping_config is not None:
            early_stopping = EarlyStopping(
                metric=early_stopping_config.get("metric", "loss"),
                min_improvement=early_stopping_config.get("min_improvement", 0),
                steps=early_stopping_config["steps"],
            )
        else:
            early_stopping = None
        return cls(
            model,
            features_file or config["data"]["eval_features_file"],
            labels_file or config["data"].get("eval_labels_file"),
            eval_config["batch_size"],
            batch_type=eval_config.get("batch_type", "examples"),
            length_bucket_width=eval_config.get("length_bucket_width"),
            scorers=scorers,
            save_predictions=eval_config.get("save_eval_predictions", False),
            early_stopping=early_stopping,
            model_dir=config["model_dir"],
            export_on_best=eval_config.get("export_on_best"),
            exporter=exporters.make_exporter(
                eval_config.get("export_format", "saved_model")
            ),
            max_exports_to_keep=eval_config.get("max_exports_to_keep", 5),
        )

    @property
    def predictions_dir(self):
        """The directory containing saved predictions."""
        return self._eval_dir

    @property
    def export_dir(self):
        """The directory containing exported models."""
        return self._export_dir

    @property
    def metrics_name(self):
        """The name of the metrics returned by this evaluator."""
        return self._metrics_name

    @property
    def metrics_history(self):
        """The history of metrics result per evaluation step."""
        return self._metrics_history

    @property
    def last_evaluated_step(self):
        """The last training step that was evaluated."""
        if not self._metrics_history:
            return None
        return self._metrics_history[-1][0]

    def _is_higher_better_for_metric(self, metric):
        # Look if the metric is produced by a scorer as they define the scores order.
        for scorer in self._scorers:
            if metric in scorer.scores_name:
                return scorer.higher_is_better()
        # TODO: the condition below is not always true, find a way to set it
        # correctly for Keras metrics.
        return metric not in ("loss", "perplexity")

    def _get_metric_history(self, metric):
        return [
            metrics[metric] for _, metrics in self._metrics_history if metric in metrics
        ]

    def should_stop(self):
        """Returns ``True`` if early stopping conditions are met."""
        if self._early_stopping is None:
            return False
        target_metric = self._early_stopping.metric
        higher_is_better = self._is_higher_better_for_metric(target_metric)
        metrics = self._get_metric_history(target_metric)
        should_stop = early_stop(
            metrics,
            self._early_stopping.steps,
            min_improvement=self._early_stopping.min_improvement,
            higher_is_better=higher_is_better,
        )
        if should_stop:
            tf.get_logger().warning(
                "Evaluation metric '%s' did not improve more than %f in the last %d evaluations",
                target_metric,
                self._early_stopping.min_improvement,
                self._early_stopping.steps,
            )
        return should_stop

    def is_best(self, metric):
        """Returns ``True`` if the latest value of :obj:`metric` is the best so far.

        Args:
          metric: The metric to consider.

        Returns:
          A boolean.
        """
        metric_history = self._get_metric_history(metric)
        if not metric_history:
            return False
        metric_history, latest_value = metric_history[:-1], metric_history[-1]
        if not metric_history:
            return True
        if self._is_higher_better_for_metric(metric):
            return latest_value > max(metric_history)
        else:
            return latest_value < min(metric_history)

    def __call__(self, step):
        """Runs the evaluator.

        Args:
          step: The current training step.

        Returns:
          A dictionary of evaluation metrics.
        """
        tf.get_logger().info("Running evaluation for step %d", step)
        output_file = None
        output_path = None
        if self._save_predictions:
            output_path = os.path.join(self._eval_dir, "predictions.txt.%d" % step)
            output_file = tf.io.gfile.GFile(output_path, "w")
            write_fn = lambda prediction: (
                self._model.print_prediction(prediction, stream=output_file)
            )
            index_fn = lambda prediction: prediction.get("index")
            ordered_writer = misc.OrderRestorer(index_fn, write_fn)

        loss_num = 0
        loss_den = 0
        metrics = self._model.get_metrics()
        for source, target in self._dataset:
            loss, predictions = self._eval_fn(source, target)
            if isinstance(loss, tuple):
                loss_num += loss[0]
                loss_den += loss[1]
            else:
                loss_num += loss
                loss_den += 1
            if metrics:
                self._model.update_metrics(metrics, predictions, target)
            if output_file is not None:
                predictions = {k: v.numpy() for k, v in predictions.items()}
                for prediction in misc.extract_batches(predictions):
                    ordered_writer.push(prediction)
        if loss_den == 0:
            raise RuntimeError("No examples were evaluated")
        loss = loss_num / loss_den

        results = dict(loss=loss, perplexity=tf.math.exp(loss))
        if metrics:
            for name, metric in metrics.items():
                results[name] = metric.result()
        if self._save_predictions:
            tf.get_logger().info("Evaluation predictions saved to %s", output_path)
            output_file.close()
            for scorer in self._scorers:
                score = scorer(self._labels_file, output_path)
                if isinstance(score, dict):
                    results.update(score)
                else:
                    results[scorer.name] = score

        for name, value in results.items():
            if isinstance(value, tf.Tensor):
                results[name] = value.numpy()

        self._record_results(step, results)
        self._maybe_export(step, results)
        self._maybe_garbage_collect_exports()
        return results

    def _record_results(self, step, results):
        # Clear history for steps that are greater than step.
        while self._metrics_history and self._metrics_history[-1][0] > step:
            self._metrics_history.pop()
        self._metrics_history.append((step, dict(results)))
        tf.get_logger().info(
            "Evaluation result for step %d: %s",
            step,
            " ; ".join("%s = %f" % (k, v) for k, v in results.items()),
        )
        with self._summary_writer.as_default():
            for key, value in results.items():
                tf.summary.scalar("%s/%s" % (_SUMMARIES_SCOPE, key), value, step=step)
            self._summary_writer.flush()

    def _maybe_export(self, step, results):
        if self._export_on_best is None or not self.is_best(self._export_on_best):
            return
        export_dir = os.path.join(self._export_dir, str(step))
        tf.get_logger().info(
            "Exporting model to %s (best %s so far: %f)",
            export_dir,
            self._export_on_best,
            results[self._export_on_best],
        )
        self._model.export(export_dir, exporter=self._exporter)

    def _maybe_garbage_collect_exports(self):
        if self._max_exports_to_keep is None or not os.path.exists(self._export_dir):
            return
        exported_steps = list(sorted(map(int, os.listdir(self._export_dir))))
        num_exports = len(exported_steps)
        if num_exports > self._max_exports_to_keep:
            steps_to_remove = exported_steps[: num_exports - self._max_exports_to_keep]
            for step in steps_to_remove:
                shutil.rmtree(os.path.join(self._export_dir, str(step)))


def early_stop(metrics, steps, min_improvement=0, higher_is_better=False):
    """Early stopping condition.

    Args:
      metrics: A list of metric values.
      steps: Consider the improvement over this many steps.
      min_improvement: Continue if the metric improved less than this value:
      higher_is_better: Whether a higher value is better for this metric.

    Returns:
      A boolean.
    """
    if len(metrics) < steps + 1:
        return False

    def _did_improve(ref, new):
        # Returns True if new is improving on ref.
        if higher_is_better:
            return new > ref + min_improvement
        else:
            return new < ref - min_improvement

    ref_metric = metrics[-steps - 1]
    for metric in metrics[-steps:]:
        if _did_improve(ref_metric, metric):
            return False
    return True
