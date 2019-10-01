"""Hypotheses file scoring."""

import abc
import six

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Scorer(object):
  """Scores hypotheses against references."""

  def __init__(self, name):
    self._name = name

  @property
  def name(self):
    """The scorer name."""
    return self._name

  @property
  def scores_name(self):
    """The names of returned scores."""
    return {self._name}

  @abc.abstractmethod
  def __call__(self, ref_path, hyp_path):
    """Scores hypotheses.

    Args:
      ref_path: Path to the reference.
      hyp_path: Path to the hypotheses.

    Returns:
      The score or dictionary of scores.
    """
    raise NotImplementedError()

  def lower_is_better(self):
    """Returns ``True`` if a lower score is better."""
    return False

  def higher_is_better(self):
    """Returns ``True`` if a higher score is better."""
    return not self.lower_is_better()


class ROUGEScorer(Scorer):
  """ROUGE scorer based on https://github.com/pltrdy/rouge."""

  def __init__(self):
    super(ROUGEScorer, self).__init__("rouge")

  @property
  def scores_name(self):
    return {"rouge-1", "rouge-2", "rouge-l"}

  def __call__(self, ref_path, hyp_path):
    from rouge import FilesRouge
    files_rouge = FilesRouge(hyp_path, ref_path)
    rouge_scores = files_rouge.get_scores(avg=True)
    return {name:rouge_scores[name]["f"] for name in self.scores_name}


class BLEUScorer(Scorer):
  """Scorer using sacreBLEU."""

  def __init__(self):
    try:
      import sacrebleu  # pylint: disable=unused-import, unused-variable
    except ImportError:
      raise ImportError("sacreBLEU evaluator requires Python 3")
    super(BLEUScorer, self).__init__("bleu")

  def __call__(self, ref_path, hyp_path):
    from sacrebleu import corpus_bleu
    with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
      bleu = corpus_bleu(sys_stream, [ref_stream])
      return bleu.score


def make_scorers(names):
  """Returns a list of scorers.

  Args:
    names: A list of scorer names or a single name.

  Returns:
    A list of :class:`opennmt.evaluation.Scorer` instances.

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
    elif name == "rouge":
      scorer = ROUGEScorer()
    else:
      raise ValueError("No scorer associated with the name: {}".format(name))
    scorers.append(scorer)
  return scorers
