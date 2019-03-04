# -*- coding: utf-8 -*-

import os
import unittest
import shutil

import tensorflow as tf

from opennmt import Runner
from opennmt.config import load_model
from opennmt.utils import misc
from opennmt.tests import test_util


test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(test_dir, "..", "..")
test_data = os.path.join(root_dir, "testdata")


@test_util.run_tf1_only
class RunnerTest(tf.test.TestCase):

  def _getTransliterationRunner(self, base_config=None):
    model_dir = os.path.join(self.get_temp_dir(), "transliteration-aren")
    shutil.copytree(os.path.join(test_data, "transliteration-aren"), model_dir)
    config = {}
    config["model_dir"] = model_dir
    config["data"] = {
        "source_vocabulary": os.path.join(model_dir, "ar.vocab"),
        "target_vocabulary": os.path.join(model_dir, "en.vocab"),
    }
    if base_config is not None:
      config = misc.merge_dict(config, base_config)
    model = load_model(model_dir)
    runner = Runner(model, config)
    return runner

  def _makeTransliterationData(self):
    ar = [
      "آ ت ز م و ن",
      "آ ت ش ي س و ن",
      "آ ر ب ا ك ه",
      "آ ر ث ر",
      "آ ز ا",
    ]
    en = [
        "a t z m o n",
        "a c h e s o n",
        "a a r b a k k e",
        "a r t h u r",
        "a s a"
    ]
    ar_file = test_util.make_data_file(os.path.join(self.get_temp_dir(), "ar.txt"), ar)
    en_file = test_util.make_data_file(os.path.join(self.get_temp_dir(), "en.txt"), en)
    return ar_file, en_file

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testTrain(self):
    ar_file, en_file  = self._makeTransliterationData()
    config = {
        "data": {
            "train_features_file": ar_file,
            "train_labels_file": en_file
        },
        "params": {
            "learning_rate": 0.0005,
            "optimizer": "AdamOptimizer"
        },
        "train": {
            "batch_size": 10,
            "train_steps": 145001  # Just train for 1 step.
        }
    }
    runner = self._getTransliterationRunner(config)
    runner.train()

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testEvaluate(self):
    ar_file, en_file  = self._makeTransliterationData()
    config = {
        "data": {
            "eval_features_file": ar_file,
            "eval_labels_file": en_file
        },
        "eval": {
            "external_evaluators": "BLEU"
        }
    }
    runner = self._getTransliterationRunner(config)
    metrics = runner.evaluate()
    self.assertIn("loss", metrics)

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testTrainAndEvaluate(self):
    ar_file, en_file  = self._makeTransliterationData()
    config = {
        "data": {
            "train_features_file": ar_file,
            "train_labels_file": en_file,
            "eval_features_file": ar_file,
            "eval_labels_file": en_file
        },
        "params": {
            "learning_rate": 0.0005,
            "optimizer": "AdamOptimizer"
        },
        "train": {
            "batch_size": 10,
            "train_steps": 145001  # Just train for 1 step.
        }
    }
    runner = self._getTransliterationRunner(config)
    result = runner.train_and_evaluate()
    if result is not None:
      metrics, export = result
      self.assertIn("loss", metrics)
      self.assertEqual(len(export), 1)

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testInfer(self):
    runner = self._getTransliterationRunner()
    ar_file, _ = self._makeTransliterationData()
    en_file = os.path.join(self.get_temp_dir(), "output.txt")
    runner.infer(ar_file, predictions_file=en_file)
    self.assertTrue(os.path.exists(en_file))
    with open(en_file) as f:
      lines = f.readlines()
    self.assertEqual(len(lines), 5)

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testScore(self):
    runner = self._getTransliterationRunner()
    ar_file, en_file = self._makeTransliterationData()
    score_file = os.path.join(self.get_temp_dir(), "scores.txt")
    runner.score(ar_file, en_file, output_file=score_file)
    self.assertTrue(os.path.exists(score_file))
    with open(score_file) as f:
      lines = f.readlines()
    self.assertEqual(len(lines), 5)

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testExport(self):
    export_dir_base = os.path.join(self.get_temp_dir(), "export")
    runner = self._getTransliterationRunner()
    export_dir = runner.export(export_dir_base=export_dir_base)
    with self.test_session() as sess:
      meta_graph_def = tf.saved_model.loader.load(
          sess, [tf.saved_model.tag_constants.SERVING], export_dir)


if __name__ == "__main__":
  tf.test.main()
