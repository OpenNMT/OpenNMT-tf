"""Hypotheses file scoring."""

import abc
import sacrebleu

import tensorflow as tf

from rouge import FilesRouge

from opennmt.utils import misc
from opennmt.utils.fmeasure import fmeasure
from opennmt.utils.wer import wer


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


_SCORERS_REGISTRY = misc.ClassRegistry(base_class=Scorer)
register_scorer = _SCORERS_REGISTRY.register


def _get_lines(path):
    lines = []
    with tf.io.gfile.GFile(path) as f:
        for line in f:
            lines.append(line.rstrip())
    return lines


@register_scorer(name="rouge")
class ROUGEScorer(Scorer):
    """ROUGE scorer based on https://github.com/pltrdy/rouge."""

    def __init__(self):
        super().__init__("rouge")

    @property
    def scores_name(self):
        return {"rouge-1", "rouge-2", "rouge-l"}

    def __call__(self, ref_path, hyp_path):
        scorer = FilesRouge(metrics=list(self.scores_name))
        rouge_scores = scorer.get_scores(hyp_path, ref_path, avg=True)
        return {name: rouge_scores[name]["f"] for name in self.scores_name}


@register_scorer(name="bleu")
class BLEUScorer(Scorer):
    """Scorer using sacreBLEU."""

    def __init__(self):
        super().__init__("bleu")

    def __call__(self, ref_path, hyp_path):
        sys_stream = _get_lines(hyp_path)
        ref_stream = _get_lines(ref_path)
        bleu = sacrebleu.corpus_bleu(sys_stream, [ref_stream], force=True)
        return bleu.score


@register_scorer(name="wer")
class WERScorer(Scorer):
    """WER scorer."""

    def __init__(self):
        super().__init__("wer")

    def __call__(self, ref_path, hyp_path):
        wer_score = wer(ref_path, hyp_path)
        return wer_score

    def lower_is_better(self):
        return True


@register_scorer(name="ter")
class TERScorer(Scorer):
    """TER scorer."""

    def __init__(self):
        super().__init__("ter")

    def __call__(self, ref_path, hyp_path):
        sys_stream = _get_lines(hyp_path)
        ref_stream = _get_lines(ref_path)
        ter = sacrebleu.corpus_ter(sys_stream, [ref_stream])
        return ter.score

    def lower_is_better(self):
        return True


@register_scorer(name="prfmeasure", alias="prf")
class PRFScorer(Scorer):
    """F-measure scorer."""

    def __init__(self):
        super().__init__("prfmeasure")

    @property
    def scores_name(self):
        return {"precision", "recall", "fmeasure"}

    def __call__(self, ref_path, hyp_path):
        precision_score, recall_score, fmeasure_score = fmeasure(ref_path, hyp_path)
        return {
            "precision": precision_score,
            "recall": recall_score,
            "fmeasure": fmeasure_score,
        }


def make_scorers(names):
    """Returns a list of scorers.

    Args:
      names: A list of scorer names or a single name.

    Returns:
      A list of :class:`opennmt.utils.Scorer` instances.

    Raises:
      ValueError: if a scorer name is invalid.
    """
    if not isinstance(names, list):
        names = [names]
    scorers = []
    for name in names:
        scorer_class = _SCORERS_REGISTRY.get(name.lower())
        if scorer_class is None:
            raise ValueError("No scorer associated with the name: {}".format(name))
        scorers.append(scorer_class())
    return scorers
