"""Evaluation related classes and functions."""

import subprocess

import abc
import io
import re
import os
import six

import tensorflow as tf

from opennmt.utils.misc import get_third_party_dir


@six.add_metaclass(abc.ABCMeta)
class ExternalEvaluator(object):
  """Base class for external evaluators."""

  def __init__(self, labels_file=None, output_dir=None):
    self._labels_file = labels_file
    self._summary_writer = None

    if output_dir is not None:
      self._summary_writer = tf.summary.FileWriterCache.get(output_dir)

  def __call__(self, step, predictions_path):
    """Scores the predictions and logs the result.

    Args:
      step: The step at which this evaluation occurs.
      predictions_path: The path to the saved predictions.
    """
    score = self.score(self._labels_file, predictions_path)
    if score is None:
      return
    if self._summary_writer is not None:
      self._summarize_score(step, score)
    self._log_score(score)

  def _summarize_value(self, step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    self._summary_writer.add_summary(summary, step)

  # Some evaluators may return several scores so let them the ability to
  # define how to log the score result.

  def _summarize_score(self, step, score):
    self._summarize_value(step, "external_evaluation/{}".format(self.name()), score)

  def _log_score(self, score):
    tf.logging.info("%s evaluation score: %f", self.name(), score)

  @abc.abstractproperty
  def name(self):
    """Returns the name of this evaluator."""
    raise NotImplementedError()

  @abc.abstractmethod
  def score(self, labels_file, predictions_path):
    """Scores the predictions against the true output labels."""
    raise NotImplementedError()


class ROUGEEvaluator(ExternalEvaluator):
  """ROUGE evaluator based on https://github.com/pltrdy/rouge."""

  def __init__(self, labels_file=None, output_dir=None):
    try:
      import rouge  # pylint: disable=unused-variable
    except ImportError:
      raise ImportError("Please install the 'rouge' package, run 'pip install rouge==0.3.0'")
    super(ROUGEEvaluator, self).__init__(labels_file=labels_file, output_dir=output_dir)

  def name(self):
    return "ROUGE"

  def _summarize_score(self, step, score):
    self._summarize_value(step, "external_evaluation/ROUGE-1", score["rouge-1"])
    self._summarize_value(step, "external_evaluation/ROUGE-2", score["rouge-2"])
    self._summarize_value(step, "external_evaluation/ROUGE-L", score["rouge-l"])

  def _log_score(self, score):
    tf.logging.info("Evaluation score: ROUGE-1 = %f; ROUGE-2 = %f; ROUGE-L = %s",
                    score["rouge-1"], score["rouge-2"], score["rouge-l"])

  def score(self, labels_file, predictions_path):
    from rouge import FilesRouge
    files_rouge = FilesRouge(predictions_path, labels_file)
    rouge_scores = files_rouge.get_scores(avg=True)
    return {k:v["f"] for k, v in six.iteritems(rouge_scores)}


class BLEUEvaluator(ExternalEvaluator):
  """Evaluator calling multi-bleu.perl."""

  def _get_bleu_script(self):
    return "multi-bleu.perl"

  def name(self):
    return "BLEU"

  def score(self, labels_file, predictions_path):
    bleu_script = self._get_bleu_script()
    try:
      third_party_dir = get_third_party_dir()
    except RuntimeError as e:
      tf.logging.warning("%s", str(e))
      return None
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [os.path.join(third_party_dir, bleu_script), labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "{} script returned non-zero exit code: {}".format(bleu_script, msg))
      return None


class BLEUDetokEvaluator(BLEUEvaluator):
  """Evaluator calling multi-bleu-detok.perl."""

  def _get_bleu_script(self):
    return "multi-bleu-detok.perl"

  def name(self):
    return "BLEU-detok"


def external_evaluation_fn(evaluators_name, labels_file, output_dir=None):
  """Returns a callable to be used in
  :class:`opennmt.utils.hooks.SaveEvaluationPredictionHook` that calls one or
  more external evaluators.

  Args:
    evaluators_name: An evaluator name or a list of evaluators name.
    labels_file: The true output labels.
    output_dir: The run directory.

  Returns:
    A callable or ``None`` if :obj:`evaluators_name` is ``None`` or empty.

  Raises:
    ValueError: if an evaluator name is invalid.
  """
  if evaluators_name is None:
    return None
  if not isinstance(evaluators_name, list):
    evaluators_name = [evaluators_name]
  if not evaluators_name:
    return None

  evaluators = []
  for name in evaluators_name:
    name = name.lower()
    if name == "bleu":
      evaluator = BLEUEvaluator(labels_file=labels_file, output_dir=output_dir)
    elif name == "bleu-detok":
      evaluator = BLEUDetokEvaluator(labels_file=labels_file, output_dir=output_dir)
    elif name == "rouge":
      evaluator = ROUGEEvaluator(labels_file=labels_file, output_dir=output_dir)
    else:
      raise ValueError("No evaluator associated with the name: {}".format(name))
    evaluators.append(evaluator)

  def _post_evaluation_fn(step, predictions_path):
    for evaluator in evaluators:
      evaluator(step, predictions_path)

  return _post_evaluation_fn
