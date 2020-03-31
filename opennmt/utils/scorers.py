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
    super(WERScorer, self).__init__("wer")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.wer import wer # pylint: disable=import-outside-toplevel
    wer_score = wer(ref_path, hyp_path)
    return wer_score

  def lower_is_better(self):
    """ Since the score shall be the lower the better """
    return True

class TERScorer(Scorer):
  """Scorer for TER."""
  def __init__(self):
    super(TERScorer, self).__init__("ter")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.ter import ter # pylint: disable=import-outside-toplevel
    ter_score = ter(ref_path, hyp_path)
    return ter_score

  def lower_is_better(self):
    """ Since the score shall be the lower the better """
    return True



class FMEASUREScorer(Scorer):
  """Scorer for F-measure."""
  def __init__(self):
    super(FMEASUREScorer, self).__init__("fmeasure")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure  # pylint: disable=import-outside-toplevel
    fmeasure_score = fmeasure(ref_path, hyp_path, False, False, True)
    return fmeasure_score

class PRECISIONScorer(Scorer):
  """Scorer for Precision."""
  def __init__(self):
    super(PRECISIONScorer, self).__init__("precision")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure  # pylint: disable=import-outside-toplevel
    precision_score = fmeasure(ref_path, hyp_path, True, False, False)
    return precision_score

class RECALLScorer(Scorer):
  """Scorer for Recall."""
  def __init__(self):
    super(RECALLScorer, self).__init__("recall")

  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure  # pylint: disable=import-outside-toplevel
    recall_score = fmeasure(ref_path, hyp_path, False, True, False)
    return recall_score

class PRFScorer(Scorer):
  """Scorer for F-measure."""
  def __init__(self):
    super(PRFScorer, self).__init__("prfmeasure")

  @property
  def scores_name(self):
    return {"precision", "recall", "fmeasure"}


  def __call__(self, ref_path, hyp_path):
    from opennmt.utils.fmeasure import fmeasure  # pylint: disable=import-outside-toplevel
    precision_score, recall_score, fmeasure_score = fmeasure(ref_path, hyp_path)
    return {"precision":precision_score, "recall":recall_score, "fmeasure":fmeasure_score}


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
    elif name == "wer":
      scorer = WERScorer()
    elif name == "ter":
      scorer = TERScorer()
    elif name == "precision":
      scorer = PRECISIONScorer()
    elif name == "recall":
      scorer = RECALLScorer()
    elif name == "fmeasure":
      scorer = FMEASUREScorer()
    elif name == "prfmeasure":
      scorer = PRFScorer()
    else:
      raise ValueError("No scorer associated with the name: {}".format(name))
    scorers.append(scorer)
  return scorers
