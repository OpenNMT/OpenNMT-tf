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


if __name__ == "__main__":
  tf.test.main()
