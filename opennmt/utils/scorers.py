"""Hypotheses file scoring."""

import abc

import tensorflow as tf


class Scorer(abc.ABC):
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
    from rouge import FilesRouge  # pylint: disable=import-outside-toplevel
    scorer = FilesRouge(metrics=list(self.scores_name))
    rouge_scores = scorer.get_scores(hyp_path, ref_path, avg=True)
    return {name:rouge_scores[name]["f"] for name in self.scores_name}


class BLEUScorer(Scorer):
  """Scorer using sacreBLEU."""

  def __init__(self):
    super(BLEUScorer, self).__init__("bleu")

  def __call__(self, ref_path, hyp_path):
    from sacrebleu import corpus_bleu  # pylint: disable=import-outside-toplevel
    with tf.io.gfile.GFile(ref_path) as ref_stream, tf.io.gfile.GFile(hyp_path) as sys_stream:
      bleu = corpus_bleu(sys_stream, [ref_stream], force=True)
      return bleu.score

class WERScorer(Scorer):
  """Scorer for WER."""
  def __init__(self):
    super(WERScorer, self).__init__("iwer")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.wer import wer, sentence_wer
    wer_score=wer(ref_path,hyp_path)
    return (1.0-wer_score)

class FMEASUREScorer(Scorer):
  """Scorer for F-measure."""
  def __init__(self):
    super(FMEASUREScorer, self).__init__("fmeasure")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure
    precision_score,recall_score,fmeasure_score=fmeasure(ref_path,hyp_path)
    return fmeasure_score

class PRECISIONScorer(Scorer):
  """Scorer for Precision."""
  def __init__(self):
    super(PRECISIONScorer, self).__init__("fmeasure")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure
    precision_score,recall_score,fmeasure_score=fmeasure(ref_path,hyp_path)
    return precision_score

class RECALLScorer(Scorer):
  """Scorer for Recall."""
  def __init__(self):
    super(RECALLScorer, self).__init__("fmeasure")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure
    precision_score,recall_score,fmeasure_score=fmeasure(ref_path,hyp_path)
    return recall_score



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
    elif name == "iwer":
      scorer = WERScorer()
    elif name == "precision":
      scorer = PRECISIONScorer()
    elif name == "recall":
      scorer = RECALLScorer()
    elif name == "fmeasure":
      scorer = FMEASUREScorer()
    else:
      raise ValueError("No scorer associated with the name: {}".format(name))
    scorers.append(scorer)
  return scorers
