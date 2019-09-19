import os
import six

import tensorflow as tf

from opennmt import evaluation
from opennmt import inputters
from opennmt import models


# Define some dummy components to simply return the loss and metrics we want to test.

class TestMetric(object):
  def __init__(self, result_history):
    self.result_history = result_history

  def result(self):
    result = self.result_history[0]
    self.result_history.pop(0)
    return result

class TestInputter(inputters.Inputter):
  def make_dataset(self, data_file, training=None):
    return tf.data.TextLineDataset(data_file)

  def input_signature(self):
    return None

  def make_features(self, element=None, features=None, training=None):
    return tf.strings.to_number(element)

class TestModel(models.Model):
  def __init__(self, loss_history, metrics_history):
    example_inputter = inputters.ExampleInputter(TestInputter(), TestInputter())
    super(TestModel, self).__init__(example_inputter)
    self.loss_history = loss_history
    self.metrics_history = metrics_history

  def call(self, features, labels=None, training=None, step=None):
    return features, None

  def get_metrics(self):
    return {name:TestMetric(history) for name, history in self.metrics_history.items()}

  def compute_loss(self, outputs, labels, training=True):
    loss = self.loss_history[0]
    self.loss_history.pop(0)
    return loss


class EvaluationTest(tf.test.TestCase):

  def testEvaluationMetric(self):
    features_file = os.path.join(self.get_temp_dir(), "features.txt")
    labels_file = os.path.join(self.get_temp_dir(), "labels.txt")
    eval_dir = os.path.join(self.get_temp_dir(), "eval")
    with open(features_file, "w") as features, open(labels_file, "w") as labels:
      features.write("1\n2\n")
      labels.write("1\n2\n")
    model = TestModel([1, 4, 7], {"a": [2, 5, 8], "b": [3, 6, 9]})
    model.initialize({})
    early_stopping = evaluation.EarlyStopping(metric="loss", min_improvement=0, steps=1)
    evaluator = evaluation.Evaluator(
        model,
        features_file,
        labels_file,
        batch_size=1,
        early_stopping=early_stopping,
        eval_dir=eval_dir)
    self.assertSetEqual(evaluator.metrics_name, {"loss", "a", "b"})
    metrics_5 = evaluator(5)
    self.assertDictEqual(metrics_5, {"loss": 1.0, "a": 2, "b": 3})
    self.assertFalse(evaluator.should_stop())
    metrics_10 = evaluator(10)
    self.assertDictEqual(metrics_10, {"loss": 4.0, "a": 5, "b": 6})
    self.assertTrue(evaluator.should_stop())
    self.assertLen(evaluator.metrics_history, 2)
    self.assertEqual(evaluator.metrics_history[0][0], 5)
    self.assertEqual(evaluator.metrics_history[0][1], metrics_5)
    self.assertEqual(evaluator.metrics_history[1][0], 10)
    self.assertEqual(evaluator.metrics_history[1][1], metrics_10)

    # Recreating the evaluator should load the metrics history from the eval directory.
    evaluator = evaluation.Evaluator(
        model,
        features_file,
        labels_file,
        batch_size=1,
        eval_dir=eval_dir)
    self.assertLen(evaluator.metrics_history, 2)
    self.assertEqual(evaluator.metrics_history[0][0], 5)
    self.assertEqual(evaluator.metrics_history[0][1], metrics_5)
    self.assertEqual(evaluator.metrics_history[1][0], 10)
    self.assertEqual(evaluator.metrics_history[1][1], metrics_10)

    # Evaluating previous steps should clear future steps in the history.
    self.assertDictEqual(evaluator(7), {"loss": 7.0, "a": 8, "b": 9})
    recorded_steps = list(step for step, _ in evaluator.metrics_history)
    self.assertListEqual(recorded_steps, [5, 7])

  def testEarlyStop(self):
    self.assertFalse(
        evaluation.early_stop([3.1, 2.7, 2.6], 4))
    self.assertFalse(
        evaluation.early_stop([3.1, 2.7, 2.6, 2.5], 4))
    self.assertTrue(
        evaluation.early_stop([3.1, 3.1, 3.0, 2.9], 3, min_improvement=0.3))
    self.assertTrue(
        evaluation.early_stop([32, 33, 32, 33, 32], 3, min_improvement=2, higher_is_better=False))
    self.assertFalse(
        evaluation.early_stop([32, 35, 32, 33, 32], 3, min_improvement=2, higher_is_better=False))


if __name__ == "__main__":
  tf.test.main()
