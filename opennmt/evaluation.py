"""Evaluation related classes and functions."""

import collections
import os
import six

import tensorflow as tf

from opennmt.data import dataset as dataset_lib
from opennmt.utils import misc
from opennmt.utils import scorers as scorers_lib


_SUMMARIES_SCOPE = "metrics"


class EarlyStopping(
    collections.namedtuple("EarlyStopping",
                           ("metric", "min_improvement", "steps"))):
  """Conditions for early stopping."""


class Evaluator(object):
  """Model evaluator."""

  def __init__(self,
               model,
               features_file,
               labels_file,
               batch_size,
               scorers=None,
               save_predictions=False,
               early_stopping=None,
               eval_dir=None):
    """Initializes the evaluator.

    Args:
      model: A :class:`opennmt.models.model.Model` to evaluate.
      features_file: Path to the evaluation features.
      labels_file: Path to the evaluation labels.
      batch_size: The evaluation batch size.
      scorers: A list of scorers, callables taking the path to the reference and
        the hypothesis and return one or more scores.
      save_predictions: Save evaluation predictions to a file. This is ``True``
        when :obj:`external_evaluator` is set.
      early_stopping: An ``EarlyStopping`` instance.
      eval_dir: Directory where predictions can be saved.

    Raises:
      ValueError: If predictions should be saved but the model is not compatible.
      ValueError: If predictions should be saved but :obj:`eval_dir` is ``None``.
      ValueError: If the :obj:`early_stopping` configuration is invalid.
    """
    if scorers is None:
      scorers = []
    if scorers:
      save_predictions = True
    if save_predictions:
      if model.unsupervised:
        raise ValueError("This model does not support saving evaluation predictions")
      if eval_dir is None:
        raise ValueError("Saving evaluation predictions requires eval_dir to be set")
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
        num_threads=1,
        prefetch_buffer_size=1)

    @dataset_lib.function_on_next(dataset)
    def _eval(next_fn):
      source, target = next_fn()
      outputs, predictions = model(source, labels=target)
      loss = model.compute_loss(outputs, target, training=False)
      return loss, predictions, target

    self._eval = _eval

    self._metrics_name = {"loss", "perplexity"}
    for scorer in self._scorers:
      self._metrics_name.update(scorer.scores_name)
    model_metrics = self._model.get_metrics()
    if model_metrics:
      self._metrics_name.update(set(six.iterkeys(model_metrics)))

    if early_stopping is not None:
      if early_stopping.metric not in self._metrics_name:
        raise ValueError("Invalid early stopping metric '%s', expected one in %s" % (
            early_stopping.metric, str(self._metrics_name)))
      if early_stopping.steps <= 0:
        raise ValueError("Early stopping steps should greater than 0")
    self._early_stopping = early_stopping

  @classmethod
  def from_config(cls, model, config, features_file=None, labels_file=None):
    """Creates an evaluator from the configuration.

    Args:
      model: A :class:`opennmt.models.model.Model` to evaluate.
      config: The global user configuration.
      features_file: Optional input features file to evaluate. If not set, will
        load ``eval_features_file`` from the data configuration.
      labels_file: Optional output labels file to evaluate. If not set, will load
        ``eval_labels_file`` from the data configuration.

    Returns:
      A :class:`opennmt.evaluation.Evaluator` instance.

    Raises:
      ValueError: if one of :obj:`features_file` and :obj:`labels_file` is set
        but not the other.
    """
    if (features_file is None) != (labels_file is None):
      raise ValueError("features_file and labels_file should be both set for evaluation")
    scorers = config["eval"].get("external_evaluators")
    if scorers is not None:
      scorers = scorers_lib.make_scorers(scorers)
    early_stopping_config = config["eval"].get("early_stopping")
    if early_stopping_config is not None:
      early_stopping = EarlyStopping(
          metric=early_stopping_config.get("metric", "loss"),
          min_improvement=early_stopping_config.get("min_improvement", 0),
          steps=early_stopping_config["steps"])
    else:
      early_stopping = None
    return cls(
        model,
        features_file or config["data"]["eval_features_file"],
        labels_file or config["data"].get("eval_labels_file"),
        config["eval"]["batch_size"],
        scorers=scorers,
        save_predictions=config["eval"].get("save_eval_predictions", False),
        early_stopping=early_stopping,
        eval_dir=os.path.join(config["model_dir"], "eval"))

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
        metrics[metric] for _, metrics in self._metrics_history if metric in metrics]

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
        higher_is_better=higher_is_better)
    if should_stop:
      tf.get_logger().warning(
          "Evaluation metric '%s' did not improve more than %f in the last %d evaluations",
          target_metric,
          self._early_stopping.min_improvement,
          self._early_stopping.steps)
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

    loss_num = 0
    loss_den = 0
    metrics = self._model.get_metrics()
    for loss, predictions, target in self._eval():  # pylint: disable=no-value-for-parameter
      if isinstance(loss, tuple):
        loss_num += loss[0]
        loss_den += loss[1]
      else:
        loss_num += loss
        loss_den += 1
      if metrics:
        self._model.update_metrics(metrics, predictions, target)
      if output_file is not None:
        predictions = {k:v.numpy() for k, v in six.iteritems(predictions)}
        for prediction in misc.extract_batches(predictions):
          self._model.print_prediction(prediction, stream=output_file)
    if loss_den == 0:
      raise RuntimeError("No examples were evaluated")
    loss = loss_num / loss_den

    results = dict(loss=loss, perplexity=tf.math.exp(loss))
    if metrics:
      for name, metric in six.iteritems(metrics):
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

    return self._record_results(step, results)

  def _record_results(self, step, results):
    for name, value in six.iteritems(results):
      if isinstance(value, tf.Tensor):
        results[name] = value.numpy()
    # Clear history for steps that are greater than step.
    while self._metrics_history and self._metrics_history[-1][0] > step:
      self._metrics_history.pop()
    self._metrics_history.append((step, dict(results)))
    tf.get_logger().info(
        "Evaluation result for step %d: %s",
        step,
        " ; ".join("%s = %f" % (k, v) for k, v in six.iteritems(results)))
    with self._summary_writer.as_default():
      for key, value in six.iteritems(results):
        tf.summary.scalar("%s/%s" % (_SUMMARIES_SCOPE, key), value, step=step)
      self._summary_writer.flush()
    return results


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
