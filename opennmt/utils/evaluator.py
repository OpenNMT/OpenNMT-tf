"""Evaluation related classes and functions."""

import subprocess

import abc
import re
import six

import tensorflow as tf

from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache
from opennmt import tokenizers


def _word_level_tokenization(input_filename, output_filename):
  tokenizer = tokenizers.OpenNMTTokenizer()
  with open(input_filename, "rb") as input_file, open(output_filename, "wb") as output_file:
    tokenizer.tokenize_stream(input_stream=input_file, output_stream=output_file)


@six.add_metaclass(abc.ABCMeta)
class ExternalEvaluator(object):
  """Base class for external evaluators."""

  def __init__(self, labels_file=None, output_dir=None):
    self._labels_file = labels_file
    self._summary_writer = None

    if output_dir is not None:
      self._summary_writer = SummaryWriterCache.get(output_dir)

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

  # Some evaluators may return several scores so let them the ability to
  # define how to log the score result.

  def _summarize_score(self, step, score):
    summary = tf.Summary(value=[tf.Summary.Value(
        tag="external_evaluation/{}".format(self.name()), simple_value=score)])
    self._summary_writer.add_summary(summary, step)

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


class BLEUEvaluator(ExternalEvaluator):
  """Evaluator calling multi-bleu.perl."""

  def name(self):
    return "BLEU"

  def score(self, labels_file, predictions_path):
    try:
      with open(predictions_path, "r") as predictions_file:
        bleu_out = subprocess.check_output(
            ["third_party/multi-bleu.perl", labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "multi-bleu.perl script returned non-zero exit code: {}".format(msg))
      return None


class BLEUDetokEvaluator(BLEUEvaluator):
  """Evaluator applying a simple tokenization before calling multi-bleu.perl."""

  def __init__(self, labels_file=None, output_dir=None):
    if not hasattr(tokenizers, "OpenNMTTokenizer"):
      raise RuntimeError("The BLEU-detok evaluator only works when the OpenNMT tokenizer "
                         "is available. Please re-check its installation.")
    super(BLEUDetokEvaluator, self).__init__(labels_file=labels_file, output_dir=output_dir)

  def name(self):
    return "BLEU-detok"

  def score(self, labels_file, predictions_path):
    tok_labels_file = labels_file + ".light_tok"
    tok_predictions_path = predictions_path + ".light_tok"
    _word_level_tokenization(labels_file, tok_labels_file)
    _word_level_tokenization(predictions_path, tok_predictions_path)
    return super(BLEUDetokEvaluator, self).score(tok_labels_file, tok_predictions_path)


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
    else:
      raise ValueError("No evaluator associated with the name: {}".format(name))
    evaluators.append(evaluator)

  def _post_evaluation_fn(step, predictions_path):
    for evaluator in evaluators:
      evaluator(step, predictions_path)

  return _post_evaluation_fn
