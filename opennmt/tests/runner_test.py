# -*- coding: utf-8 -*-

import copy
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


class RunnerTest(tf.test.TestCase):

  def _getTransliterationRunner(self, base_config=None):
    model_dir = os.path.join(self.get_temp_dir(), "transliteration-aren-v1")
    shutil.copytree(os.path.join(test_data, "transliteration-aren-v1"), model_dir)
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
            "optimizer": "Adam"
        },
        "train": {
            "batch_size": 10,
            "max_step": 145001  # Just train for 1 step.
        }
    }
    runner = self._getTransliterationRunner(config)
    output_dir = runner.train()
    self.assertTrue(tf.train.latest_checkpoint(output_dir).endswith("145001"))

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
    self.assertIn("bleu", metrics)

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
    self.assertEqual(lines[0].strip(), "a t z m o n")

  @unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
  def testUpdateVocab(self):
    config = {
        "params": {
            "learning_rate": 0.0005,
            "optimizer": "Adam"
        }
    }
    runner = self._getTransliterationRunner(config)

    # Reverse order of non special tokens.
    new_en_vocab = os.path.join(self.get_temp_dir(), "en.vocab.new")
    with open(os.path.join(runner._config["model_dir"], "en.vocab")) as en_vocab, \
         open(new_en_vocab, "w") as new_vocab:
      tokens = en_vocab.readlines()
      for token in tokens[:3]:
        new_vocab.write(token)
      for token in reversed(tokens[3:]):
        new_vocab.write(token)

    output_dir = os.path.join(self.get_temp_dir(), "updated_vocab")
    self.assertEqual(runner.update_vocab(output_dir, tgt_vocab=new_en_vocab), output_dir)

    # Check that the translation is unchanged.
    new_config = copy.deepcopy(runner._config)
    new_config["model_dir"] = output_dir
    new_config["data"]["target_vocabulary"] = new_en_vocab
    runner = Runner(runner._model, new_config)
    ar_file, _ = self._makeTransliterationData()
    en_file = os.path.join(self.get_temp_dir(), "output.txt")
    runner.infer(ar_file, predictions_file=en_file)
    with open(en_file) as f:
      self.assertEqual(next(f).strip(), "a t z m o n")

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
    export_dir = os.path.join(self.get_temp_dir(), "export")
    runner = self._getTransliterationRunner()
    runner.export(export_dir)
    self.assertTrue(tf.saved_model.contains_saved_model(export_dir))


if __name__ == "__main__":
  tf.test.main()
