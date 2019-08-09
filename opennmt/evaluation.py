"""Evaluation related classes and functions."""

import os
import six

import tensorflow as tf

from opennmt.data import dataset as dataset_lib
from opennmt.utils import misc
from opennmt.utils import scorers as scorers_lib


class Evaluator(object):
  """Model evaluator."""

  def __init__(self,
               model,
               features_file,
               labels_file,
               batch_size,
               scorers=None,
               save_predictions=False,
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
      eval_dir: Directory where predictions can be saved.

    Raises:
      ValueError: If predictions should be saved but the model is not compatible.
      ValueError: If predictions should be saved but :obj:`eval_dir` is ``None``.
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
    if eval_dir is not None:
      self._summary_writer = tf.summary.create_file_writer(eval_dir)
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
    return cls(
        model,
        features_file or config["data"]["eval_features_file"],
        labels_file or config["data"].get("eval_labels_file"),
        config["eval"]["batch_size"],
        scorers=scorers,
        save_predictions=config["eval"].get("save_eval_predictions", False),
        eval_dir=os.path.join(config["model_dir"], "eval"))

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
    loss = loss_num / loss_den

    results = dict(loss=loss)
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

    for name, value in six.iteritems(results):
      if isinstance(value, tf.Tensor):
        results[name] = value.numpy()
    tf.get_logger().info(
        "Evaluation result for step %d: %s",
        step,
        " ; ".join("%s = %f" % (k, v) for k, v in six.iteritems(results)))
    with self._summary_writer.as_default():
      for key, value in six.iteritems(results):
        tf.summary.scalar("metrics/%s" % key, value, step=step)
    return results