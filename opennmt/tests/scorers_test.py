import os

import tensorflow as tf

from opennmt.tests import test_util
from opennmt.utils import scorers


class ScorersTest(tf.test.TestCase):
    def _run_scorer(self, scorer, refs, hyps):
        ref_path = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "ref.txt"), refs
        )
        hyp_path = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "hyp.txt"), hyps
        )
        return scorer(ref_path, hyp_path)

    def testBLEUScorer(self):
        refs = ["Hello world !", "How is it going ?"]
        scorer = scorers.BLEUScorer()
        score = self._run_scorer(scorer, refs, refs)
        self.assertEqual(100, int(score))

    def testROUGEScorer(self):
        refs = ["Hello world !", "How is it going ?"]
        scorer = scorers.ROUGEScorer()
        score = self._run_scorer(scorer, refs, refs)
        self.assertIsInstance(score, dict)
        self.assertIn("rouge-l", score)
        self.assertIn("rouge-1", score)
        self.assertIn("rouge-2", score)
        self.assertAlmostEqual(1.0, score["rouge-1"])

    def testWERScorer(self):
        refs = ["Hello world !", "How is it going ?"]
        scorer = scorers.WERScorer()
        score = self._run_scorer(scorer, refs, refs)
        self.assertEqual(score, 0)

    def testTERScorer(self):
        refs = ["Hello world !", "How is it going ?"]
        scorer = scorers.TERScorer()
        score = self._run_scorer(scorer, refs, refs)
        self.assertEqual(score, 0)

    def testPRFScorer(self):
        scorer = scorers.PRFScorer()
        score = self._run_scorer(
            scorer, refs=["TAG O TAG O O TAG TAG"], hyps=["TAG O O O TAG TAG O"]
        )
        expected_precision = 2 / 3
        expected_recall = 2 / 4
        expected_fscore = (
            2
            * (expected_precision * expected_recall)
            / (expected_precision + expected_recall)
        )
        self.assertAlmostEqual(score["precision"], expected_precision, places=6)
        self.assertAlmostEqual(score["recall"], expected_recall, places=6)
        self.assertAlmostEqual(score["fmeasure"], expected_fscore, places=6)

    def testPRFScorerEmptyLine(self):
        scorer = scorers.PRFScorer()
        self._run_scorer(scorer, [""], ["O TAG"])
        self._run_scorer(scorer, ["O TAG"], [""])

    def testMakeScorers(self):
        def _check_scorers(scorers, instances):
            self.assertLen(scorers, len(instances))
            for scorer, instance in zip(scorers, instances):
                self.assertIsInstance(scorer, instance)

        _check_scorers(scorers.make_scorers("bleu"), [scorers.BLEUScorer])
        _check_scorers(scorers.make_scorers("BLEU"), [scorers.BLEUScorer])
        _check_scorers(
            scorers.make_scorers(["BLEU", "rouge"]),
            [scorers.BLEUScorer, scorers.ROUGEScorer],
        )
        _check_scorers(scorers.make_scorers("prf"), [scorers.PRFScorer])
        _check_scorers(scorers.make_scorers("prfmeasure"), [scorers.PRFScorer])


if __name__ == "__main__":
    tf.test.main()
