import tensorflow as tf

from opennmt.utils import evaluator


class EvaluatorTest(tf.test.TestCase):

  def testBLEUEvaluator(self):
    bleu_evaluator = evaluator.BLEUEvaluator()
    score = bleu_evaluator.score("data/toy-ende/tgt-val.txt", "data/toy-ende/tgt-val.txt")
    self.assertEqual(100.0, score)

  def testBLEUDetokEvaluator(self):
    bleu_evaluator = evaluator.BLEUDetokEvaluator()
    score = bleu_evaluator.score("data/toy-ende/tgt-val.txt", "data/toy-ende/tgt-val.txt")
    self.assertEqual(100.0, score)

  def testROUGEEvaluator(self):
    rouge_evaluator = evaluator.ROUGEEvaluator()
    score = rouge_evaluator.score("data/toy-ende/tgt-val.txt", "data/toy-ende/tgt-val.txt")
    self.assertIsInstance(score, dict)
    self.assertIn("rouge-l", score)
    self.assertIn("rouge-1", score)
    self.assertIn("rouge-2", score)
    self.assertAlmostEqual(1.0, score["rouge-1"])


if __name__ == "__main__":
  tf.test.main()
