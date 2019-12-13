import os

import tensorflow as tf

from opennmt.utils import scorers


class ScorersTest(tf.test.TestCase):

  def _make_perfect_hypothesis_file(self):
    ref_path = os.path.join(self.get_temp_dir(), "ref.txt")
    hyp_path = os.path.join(self.get_temp_dir(), "hyp.txt")
    with open(ref_path, "w") as ref_file, open(hyp_path, "w") as hyp_file:
      text = "Hello world !\nHow is it going ?\n"
      ref_file.write(text)
      hyp_file.write(text)
    return ref_path, hyp_path

  def testBLEUScorer(self):
    bleu_scorer = scorers.BLEUScorer()
    ref_path, hyp_path = self._make_perfect_hypothesis_file()
    score = bleu_scorer(ref_path, hyp_path)
    self.assertEqual(100, int(score))

  def testROUGEScorer(self):
    rouge_scorer = scorers.ROUGEScorer()
    ref_path, hyp_path = self._make_perfect_hypothesis_file()
    score = rouge_scorer(ref_path, hyp_path)
    self.assertIsInstance(score, dict)
    self.assertIn("rouge-l", score)
    self.assertIn("rouge-1", score)
    self.assertIn("rouge-2", score)
    self.assertAlmostEqual(1.0, score["rouge-1"])


if __name__ == "__main__":
  tf.test.main()
