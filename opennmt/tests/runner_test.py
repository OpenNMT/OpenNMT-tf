# -*- coding: utf-8 -*-

import copy
import os
import unittest
import shutil

from parameterized import parameterized

import tensorflow as tf

from opennmt import Runner
from opennmt.config import load_model
from opennmt.utils import misc
from opennmt.tests import test_util


test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(test_dir, "..", "..")
test_data = os.path.join(root_dir, "testdata")


@unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
class RunnerTest(tf.test.TestCase):

  def _getTransliterationRunner(self, base_config=None, model_version="v2"):
    model_dir = os.path.join(self.get_temp_dir(), "model")
    shutil.copytree(os.path.join(test_data, "transliteration-aren-v2", model_version), model_dir)
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
            "average_last_checkpoints": 4,
            "save_checkpoints_steps": 1,
            "max_step": 145002  # Just train for 2 steps.
        }
    }
    runner = self._getTransliterationRunner(config)
    avg_dir = runner.train()
    self.assertEndsWith(tf.train.latest_checkpoint(avg_dir), "145002")
    self.assertLen(tf.train.get_checkpoint_state(avg_dir).all_model_checkpoint_paths, 1)
    model_dir = os.path.dirname(avg_dir)
    self.assertEndsWith(tf.train.latest_checkpoint(model_dir), "145002")
    self.assertLen(tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths, 3)

    # Check that the averaged checkpoint is usable.
    ar_file, _ = self._makeTransliterationData()
    en_file = os.path.join(self.get_temp_dir(), "output.txt")
    runner.infer(ar_file, predictions_file=en_file, checkpoint_path=avg_dir)
    with open(en_file) as f:
      self.assertEqual(next(f).strip(), "a t z m o n")

  @test_util.new_context
  def testTrainDistribute(self):
    physical_devices = tf.config.experimental.list_physical_devices("CPU")
    tf.config.experimental.set_virtual_device_configuration(
        physical_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(),
         tf.config.experimental.VirtualDeviceConfiguration()])

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
            "batch_size": 2,
            "length_bucket_width": None,
            "max_step": 145003,
            "single_pass": True,  # Test we do not fail when a batch is missing for a replica.
        }
    }
    runner = self._getTransliterationRunner(config)
    runner.train(num_devices=2)

  def testTrainWithEval(self):
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
            "optimizer": "Adam"
        },
        "train": {
            "batch_size": 10,
            "max_step": 145002  # Just train for 2 steps.
        },
        "eval": {
            "export_on_best": "loss"
        }
    }
    runner = self._getTransliterationRunner(config)
    model_dir = runner.train(with_eval=True)
    export_dir = os.path.join(model_dir, "export", "145002")
    self.assertTrue(os.path.exists(export_dir))
    self.assertTrue(tf.saved_model.contains_saved_model(export_dir))

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

  @parameterized.expand([[1, "v2"], [4, "v2"], [1, "v1"]])
  def testInfer(self, beam_size, model_version):
    config = {
        "params": {
            "beam_width": beam_size
        }
    }
    runner = self._getTransliterationRunner(config, model_version)
    ar_file, _ = self._makeTransliterationData()
    en_file = os.path.join(self.get_temp_dir(), "output.txt")
    runner.infer(ar_file, predictions_file=en_file)
    self.assertTrue(os.path.exists(en_file))
    with open(en_file) as f:
      lines = f.readlines()
    self.assertEqual(len(lines), 5)
    self.assertEqual(lines[0].strip(), "a t z m o n")

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

  def testScore(self):
    runner = self._getTransliterationRunner()
    ar_file, en_file = self._makeTransliterationData()
    score_file = os.path.join(self.get_temp_dir(), "scores.txt")
    runner.score(ar_file, en_file, output_file=score_file)
    self.assertTrue(os.path.exists(score_file))
    with open(score_file) as f:
      lines = f.readlines()
    self.assertEqual(len(lines), 5)

  def testExport(self):
    config = {
        "data": {
            "source_tokenization": {
                "mode": "char"
            }
        }
    }
    export_dir = os.path.join(self.get_temp_dir(), "export")
    runner = self._getTransliterationRunner(config)
    runner.export(export_dir)
    self.assertTrue(tf.saved_model.contains_saved_model(export_dir))
    extra_assets_dir = os.path.join(export_dir, "assets.extra")
    self.assertTrue(os.path.isdir(extra_assets_dir))
    self.assertLen(os.listdir(extra_assets_dir), 1)
    imported = tf.saved_model.load(export_dir)
    translate_fn = imported.signatures["serving_default"]
    outputs = translate_fn(
        tokens=tf.constant([["آ" ,"ت" ,"ز" ,"م" ,"و" ,"ن"]]),
        length=tf.constant([6], dtype=tf.int32))
    result = tf.nest.map_structure(lambda x: x[0, 0], outputs)
    tokens = result["tokens"][:result["length"]]
    self.assertAllEqual(tokens, [b"a", b"t", b"z", b"m", b"o", b"n"])


if __name__ == "__main__":
  tf.test.main()
