import os
import sys

import tensorflow as tf

from opennmt.utils import evaluator


class EvaluatorTest(tf.test.TestCase):

  def _make_perfect_hypothesis_file(self):
    ref_path = os.path.join(self.get_temp_dir(), "ref.txt")
    hyp_path = os.path.join(self.get_temp_dir(), "hyp.txt")
    with open(ref_path, "wb") as ref_file, open(hyp_path, "wb") as hyp_file:
      text = b"Hello world !\nHow is it going ?\n"
      ref_file.write(text)
      hyp_file.write(text)
    return ref_path, hyp_path

  def testSacreBLEUEvaluator(self):
    if sys.version_info >= (3, 0):
      bleu_evaluator = evaluator.SacreBLEUEvaluator()
      ref_path, hyp_path = self._make_perfect_hypothesis_file()
      score = bleu_evaluator.score(ref_path, hyp_path)
      self.assertEqual(100, int(score))
    else:
      with self.assertRaises(ImportError):
        bleu_evaluator = evaluator.SacreBLEUEvaluator()

  def testBLEUEvaluator(self):
    bleu_evaluator = evaluator.BLEUEvaluator()
    ref_path, hyp_path = self._make_perfect_hypothesis_file()
    score = bleu_evaluator.score(ref_path, hyp_path)
    self.assertEqual(100.0, score)

  def testBLEUDetokEvaluator(self):
    bleu_evaluator = evaluator.BLEUDetokEvaluator()
    ref_path, hyp_path = self._make_perfect_hypothesis_file()
    score = bleu_evaluator.score(ref_path, hyp_path)
    self.assertEqual(100.0, score)

  def testROUGEEvaluator(self):
    rouge_evaluator = evaluator.ROUGEEvaluator()
    ref_path, hyp_path = self._make_perfect_hypothesis_file()
    score = rouge_evaluator.score(ref_path, hyp_path)
    self.assertIsInstance(score, dict)
    self.assertIn("rouge-l", score)
    self.assertIn("rouge-1", score)
    self.assertIn("rouge-2", score)
    self.assertAlmostEqual(1.0, score["rouge-1"])


if __name__ == "__main__":
  tf.test.main()
