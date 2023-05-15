import math
import os

import tensorflow as tf

from opennmt import evaluation, inputters, models
from opennmt.utils import exporters, scorers

# Define some dummy components to simply return the loss and metrics we want to test.


class _TestMetric:
    def __init__(self, result_history):
        self.result_history = result_history

    def result(self):
        result = self.result_history[0]
        self.result_history.pop(0)
        return result


class _TestInputter(inputters.Inputter):
    def make_dataset(self, data_file, training=None):
        return tf.data.TextLineDataset(data_file)

    def input_signature(self):
        return None

    def make_features(self, element=None, features=None, training=None):
        return tf.strings.to_number(element)


class _TestModel(models.Model):
    def __init__(self, metrics_history=None):
        example_inputter = inputters.ExampleInputter(_TestInputter(), _TestInputter())
        super().__init__(example_inputter)
        if metrics_history is None:
            metrics_history = {}
        self.metrics_history = metrics_history
        self.next_loss = tf.Variable(0)

    def call(self, features, labels=None, training=None, step=None):
        return features, None

    def get_metrics(self):
        return {
            name: _TestMetric(history) for name, history in self.metrics_history.items()
        }

    def compute_loss(self, outputs, labels, training=True):
        return self.next_loss


class TestExporter(exporters.Exporter):
    def _export_model(self, model, export_dir):
        tf.io.gfile.makedirs(export_dir)  # Just create an empty directory.


class EvaluationTest(tf.test.TestCase):
    def _assertMetricsEqual(self, metrics, expected):
        self.assertLen(metrics, len(expected))
        for name in expected.keys():
            self.assertIn(name, metrics)
            self.assertAllClose(metrics[name], expected[name])

    def testEvaluationMetric(self):
        features_file = os.path.join(self.get_temp_dir(), "features.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")
        model_dir = self.get_temp_dir()
        with open(features_file, "w") as features, open(labels_file, "w") as labels:
            features.write("1\n2\n")
            labels.write("1\n2\n")
        model = _TestModel({"a": [2, 5, 8], "b": [3, 6, 9]})
        early_stopping = evaluation.EarlyStopping(
            metric="loss", min_improvement=0, steps=1
        )
        evaluator = evaluation.Evaluator(
            model,
            features_file,
            labels_file,
            batch_size=1,
            early_stopping=early_stopping,
            model_dir=model_dir,
            export_on_best="loss",
            exporter=TestExporter(),
        )
        self.assertSetEqual(evaluator.metrics_name, {"loss", "perplexity", "a", "b"})
        model.next_loss.assign(1)
        metrics_5 = evaluator(5)
        self._assertMetricsEqual(
            metrics_5, {"loss": 1.0, "perplexity": math.exp(1.0), "a": 2, "b": 3}
        )
        self.assertFalse(evaluator.should_stop())
        self.assertTrue(evaluator.is_best("loss"))
        self.assertTrue(os.path.isdir(os.path.join(evaluator.export_dir, str(5))))
        model.next_loss.assign(4)
        metrics_10 = evaluator(10)
        self._assertMetricsEqual(
            metrics_10, {"loss": 4.0, "perplexity": math.exp(4.0), "a": 5, "b": 6}
        )
        self.assertTrue(evaluator.should_stop())
        self.assertFalse(evaluator.is_best("loss"))
        self.assertFalse(os.path.isdir(os.path.join(evaluator.export_dir, str(10))))
        self.assertLen(evaluator.metrics_history, 2)
        self._assertMetricsEqual(evaluator.metrics_history[0][1], metrics_5)
        self._assertMetricsEqual(evaluator.metrics_history[1][1], metrics_10)

        # Recreating the evaluator should load the metrics history from the eval directory.
        evaluator = evaluation.Evaluator(
            model,
            features_file,
            labels_file,
            batch_size=1,
            model_dir=model_dir,
            export_on_best="loss",
            exporter=TestExporter(),
        )
        self.assertLen(evaluator.metrics_history, 2)
        self._assertMetricsEqual(evaluator.metrics_history[0][1], metrics_5)
        self._assertMetricsEqual(evaluator.metrics_history[1][1], metrics_10)

        # Evaluating previous steps should clear future steps in the history.
        model.next_loss.assign(7)
        self._assertMetricsEqual(
            evaluator(7), {"loss": 7.0, "perplexity": math.exp(7.0), "a": 8, "b": 9}
        )
        self.assertFalse(evaluator.is_best("loss"))
        self.assertFalse(os.path.isdir(os.path.join(evaluator.export_dir, str(10))))
        recorded_steps = list(step for step, _ in evaluator.metrics_history)
        self.assertListEqual(recorded_steps, [5, 7])

    def testEvaluationWithRougeScorer(self):
        features_file = os.path.join(self.get_temp_dir(), "features.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")
        model_dir = self.get_temp_dir()
        with open(features_file, "w") as features, open(labels_file, "w") as labels:
            features.write("1\n2\n")
            labels.write("1\n2\n")
        model = _TestModel()
        evaluator = evaluation.Evaluator(
            model,
            features_file,
            labels_file,
            batch_size=1,
            scorers=[scorers.ROUGEScorer()],
            model_dir=model_dir,
        )
        self.assertNotIn("rouge", evaluator.metrics_name)
        self.assertIn("rouge-1", evaluator.metrics_name)
        self.assertIn("rouge-2", evaluator.metrics_name)
        self.assertIn("rouge-l", evaluator.metrics_name)

    def testExportsGarbageCollection(self):
        features_file = os.path.join(self.get_temp_dir(), "features.txt")
        labels_file = os.path.join(self.get_temp_dir(), "labels.txt")
        model_dir = self.get_temp_dir()
        with open(features_file, "w") as features, open(labels_file, "w") as labels:
            features.write("1\n2\n")
            labels.write("1\n2\n")
        model = _TestModel()
        exporter = TestExporter()
        evaluator = evaluation.Evaluator(
            model,
            features_file,
            labels_file,
            batch_size=1,
            model_dir=model_dir,
            export_on_best="loss",
            exporter=exporter,
            max_exports_to_keep=2,
        )

        # Generate some pre-existing exports.
        for step in (5, 10, 15):
            exporter.export(model, os.path.join(evaluator.export_dir, str(step)))

        def _eval_step(step, loss, expected_exported_steps):
            model.next_loss.assign(loss)
            evaluator(step)
            exported_steps = list(sorted(map(int, os.listdir(evaluator.export_dir))))
            self.assertListEqual(exported_steps, expected_exported_steps)

        _eval_step(20, 3, [15, 20])  # Exports 5 and 10 should be removed.
        _eval_step(25, 2, [20, 25])  # Export 15 should be removed.

    def testEarlyStop(self):
        self.assertFalse(evaluation.early_stop([3.1, 2.7, 2.6], 4))
        self.assertFalse(evaluation.early_stop([3.1, 2.7, 2.6, 2.5], 4))
        self.assertTrue(
            evaluation.early_stop([3.1, 3.1, 3.0, 2.9], 3, min_improvement=0.3)
        )
        self.assertTrue(
            evaluation.early_stop(
                [32, 33, 32, 33, 32], 3, min_improvement=2, higher_is_better=False
            )
        )
        self.assertFalse(
            evaluation.early_stop(
                [32, 35, 32, 33, 32], 3, min_improvement=2, higher_is_better=False
            )
        )
        self.assertTrue(
            evaluation.early_stop(
                [50.349343, 50.553991, 50.436176, 50.419565, 50.219028, 50.375434],
                4,
                min_improvement=0.01,
                higher_is_better=True,
            )
        )


if __name__ == "__main__":
    tf.test.main()
