"""Evaluation related classes and functions."""

import subprocess

import abc
import io
import re
import os
import six

import tensorflow as tf

from opennmt.utils.misc import get_third_party_dir


class ExternalEvaluator(object):
  """Base class for external evaluators."""

  def __init__(self, labels_file=None, output_dir=None, scorers=None):
    if scorers is None:
      scorers = []
    self._scorers = scorers
    self._labels_file = labels_file
    self._summary_writer = None
    if output_dir is not None:
      self._summary_writer = tf.summary.FileWriterCache.get(output_dir)

  def add_scorer(self, scorer):
    """Adds a scorer to this evaluator."""
    self._scorers.append(scorer)

  def __call__(self, step, predictions_path):
    """Scores the predictions and logs the result.

    Args:
      step: The step at which this evaluation occurs.
      predictions_path: The path to the saved predictions.
    """
    for scorer in self._scorers:
      score = scorer(self._labels_file, predictions_path)
      if score is None:
        return
      if self._summary_writer is not None:
        scorer.summarize(self._summary_writer, step, score)
      scorer.log(score)

  def score(self, labels_file, predictions_path):
    """Kept for backward compatibility: calls the first scorer."""
    return self._scorers[0](labels_file, predictions_path)


class Scorer(object):
  """Scores hypotheses against references."""

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    """The scorer name."""
    return self._name

  @abc.abstractmethod
  def __call__(self, labels_file, predictions_path):
    """Scores the predictions against the true output labels."""
    raise NotImplementedError()

  def lower_is_better(self):
    """Returns ``True`` if a lower score is better."""
    return False

  def _summarize_value(self, writer, step, tag, value):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)

  # Some scorers may return several scores so let them the ability to
  # define how to log the score result.

  def summarize(self, writer, step, score):
    """Summarizes the score in TensorBoard.

    Args:
      writer: A summary writer.
      step: The current step.
      scorer: The score to summarize.
    """
    self._summarize_value(writer, step, "external_evaluation/{}".format(self._name), score)

  def log(self, score):
    """Logs the score in the console output."""
    tf.logging.info("%s evaluation score: %f", self._name, score)


class ROUGEScorer(Scorer):
  """ROUGE scorer based on https://github.com/pltrdy/rouge."""

  def __init__(self):
    super(ROUGEScorer, self).__init__("ROUGE")

  def summarize(self, writer, step, score):
    self._summarize_value(writer, step, "external_evaluation/ROUGE-1", score["rouge-1"])
    self._summarize_value(writer, step, "external_evaluation/ROUGE-2", score["rouge-2"])
    self._summarize_value(writer, step, "external_evaluation/ROUGE-L", score["rouge-l"])

  def log(self, score):
    tf.logging.info("Evaluation score: ROUGE-1 = %f; ROUGE-2 = %f; ROUGE-L = %s",
                    score["rouge-1"], score["rouge-2"], score["rouge-l"])

  def __call__(self, labels_file, predictions_path):
    from rouge import FilesRouge
    files_rouge = FilesRouge(predictions_path, labels_file)
    rouge_scores = files_rouge.get_scores(avg=True)
    return {k:v["f"] for k, v in six.iteritems(rouge_scores)}


class SacreBLEUScorer(Scorer):
  """Scorer using sacreBLEU."""

  def __init__(self):
    try:
      import sacrebleu  # pylint: disable=unused-import, unused-variable
    except ImportError:
      raise ImportError("sacreBLEU evaluator requires Python 3")
    super(SacreBLEUScorer, self).__init__("sacreBLEU")

  def __call__(self, labels_file, predictions_path):
    from sacrebleu import corpus_bleu
    with open(labels_file) as ref_stream, open(predictions_path) as sys_stream:
      bleu = corpus_bleu(sys_stream, [ref_stream])
      return bleu.score


class BLEUScorer(Scorer):
  """Scorer calling multi-bleu.perl."""

  def __init__(self, bleu_script="multi-bleu.perl", name="BLEU"):
    super(BLEUScorer, self).__init__(name)
    self._bleu_script = bleu_script

  def __call__(self, labels_file, predictions_path):
    try:
      third_party_dir = get_third_party_dir()
    except RuntimeError as e:
      tf.logging.warning("%s", str(e))
      return None
    try:
      with io.open(predictions_path, encoding="utf-8", mode="r") as predictions_file:
        bleu_out = subprocess.check_output(
            [os.path.join(third_party_dir, self._bleu_script), labels_file],
            stdin=predictions_file,
            stderr=subprocess.STDOUT)
        bleu_out = bleu_out.decode("utf-8")
        bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        return float(bleu_score)
    except subprocess.CalledProcessError as error:
      if error.output is not None:
        msg = error.output.strip()
        tf.logging.warning(
            "{} script returned non-zero exit code: {}".format(self._bleu_script, msg))
      return None


class BLEUDetokScorer(BLEUScorer):
  """Evaluator calling multi-bleu-detok.perl."""

  def __init__(self):
    super(BLEUDetokScorer, self).__init__(
        bleu_script="multi-bleu-detok.perl", name="BLEU-detok")


def make_scorers(names):
  """Returns a list of scorers.

  Args:
    names: A list of scorer names or a single name.

  Returns:
    A list of :class:`opennmt.utils.evaluator.Scorer` instances.

  Raises:
    ValueError: if a scorer name is invalid.
  """
  if not isinstance(names, list):
    names = [names]
  scorers = []
  for name in names:
    name = name.lower()
    scorer = None
    if name == "bleu":
      scorer = BLEUScorer()
    elif name == "bleu-detok":
      scorer = BLEUDetokScorer()
    elif name == "sacrebleu":
      scorer = SacreBLEUScorer()
    elif name == "rouge":
      scorer = ROUGEScorer()
    else:
      raise ValueError("No scorer associated with the name: {}".format(name))
    scorers.append(scorer)
  return scorers

def external_evaluation_fn(evaluators_name, labels_file, output_dir=None):
  """Returns a callable to be used in
  :class:`opennmt.utils.hooks.SaveEvaluationPredictionHook` that calls one or
  more scorers.

  Args:
    evaluators_name: A scorer name or a list of scorer names.
    labels_file: The true output labels.
    output_dir: The run directory.

  Returns:
    A callable or ``None`` if :obj:`evaluators_name` is ``None`` or empty.

  Raises:
    ValueError: if an scorer name is invalid.
  """
  if not evaluators_name:
    return None
  return ExternalEvaluator(
      labels_file=labels_file,
      output_dir=output_dir,
      scorers=make_scorers(evaluators_name))


# Keep scorer-specific evaluators for backward compatibility.

class ROUGEEvaluator(ExternalEvaluator):
  """ROUGE evaluator based on https://github.com/pltrdy/rouge."""

  def __init__(self, labels_file=None, output_dir=None):
    super(ROUGEEvaluator, self).__init__(
        labels_file=labels_file, output_dir=output_dir, scorers=[ROUGEScorer()])


class SacreBLEUEvaluator(ExternalEvaluator):
  """Evaluator using sacreBLEU."""

  def __init__(self, labels_file=None, output_dir=None):
    super(SacreBLEUEvaluator, self).__init__(
        labels_file=labels_file, output_dir=output_dir, scorers=[SacreBLEUScorer()])


class BLEUEvaluator(ExternalEvaluator):
  """Evaluator calling multi-bleu.perl."""

  def __init__(self, labels_file=None, output_dir=None):
    super(BLEUEvaluator, self).__init__(
        labels_file=labels_file, output_dir=output_dir, scorers=[BLEUScorer()])


class BLEUDetokEvaluator(ExternalEvaluator):
  """Evaluator calling multi-bleu-detok.perl."""

  def __init__(self, labels_file=None, output_dir=None):
    super(BLEUDetokEvaluator, self).__init__(
        labels_file=labels_file, output_dir=output_dir, scorers=[BLEUDetokScorer()])
