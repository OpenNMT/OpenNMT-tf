import copy
import os
import shutil
import unittest

import tensorflow as tf

from parameterized import parameterized, parameterized_class

from opennmt import Runner, decoders, models
from opennmt.config import MODEL_DESCRIPTION_FILENAME, load_model, merge_config
from opennmt.tests import test_util
from opennmt.utils import exporters, misc

test_data = test_util.get_test_data_dir()


def _get_test_class_name(cls, num, params_dict):
    eager = params_dict.get("run_functions_eagerly")
    return "%s_%s" % (cls.__name__, "eager" if eager else "graph")


@unittest.skipIf(not os.path.isdir(test_data), "Missing test data directory")
@parameterized_class(
    ["run_functions_eagerly"],
    [[False], [True]],
    class_name_func=_get_test_class_name,
)
class RunnerTest(tf.test.TestCase):
    def setUp(self):
        if hasattr(self, "run_functions_eagerly"):
            tf.config.run_functions_eagerly(self.run_functions_eagerly)

    def tearDown(self):
        tf.config.run_functions_eagerly(False)

    def _getTransliterationRunner(
        self, base_config=None, model_version="v2", pass_model_builder=False
    ):
        model_dir = os.path.join(self.get_temp_dir(), "model")
        shutil.copytree(
            os.path.join(test_data, "transliteration-aren-v2", model_version), model_dir
        )
        config = {}
        config["model_dir"] = model_dir
        config["data"] = {
            "source_vocabulary": "ar.vocab",
            "target_vocabulary": "en.vocab",
        }
        if base_config is not None:
            config = merge_config(config, base_config)
        model = load_model(model_dir, as_builder=pass_model_builder)
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
        en = ["a t z m o n", "a c h e s o n", "a a r b a k k e", "a r t h u r", "a s a"]
        ar_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "ar.txt"), ar
        )
        en_file = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "en.txt"), en
        )
        return ar_file, en_file

    @parameterized.expand([[True], [False]])
    def testTrain(self, pass_model_builder):
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "data": {"train_features_file": ar_file, "train_labels_file": en_file},
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {
                "batch_size": 10,
                "average_last_checkpoints": 4,
                "save_checkpoints_steps": 1,
                "save_summary_steps": 1,
                "max_step": 145002,  # Just train for 2 steps.
            },
        }
        runner = self._getTransliterationRunner(
            config, pass_model_builder=pass_model_builder
        )
        avg_dir, summary = runner.train(return_summary=True)
        self.assertEqual(runner.model_dir, avg_dir)
        self.assertIsInstance(summary, dict)
        self.assertEndsWith(tf.train.latest_checkpoint(avg_dir), "145002")
        self.assertLen(
            tf.train.get_checkpoint_state(avg_dir).all_model_checkpoint_paths, 1
        )
        self.assertTrue(
            os.path.isfile(os.path.join(avg_dir, MODEL_DESCRIPTION_FILENAME))
        )
        model_dir = os.path.dirname(avg_dir)
        self.assertEndsWith(tf.train.latest_checkpoint(model_dir), "145002")
        self.assertLen(
            tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths, 3
        )

        # Check that the averaged checkpoint is usable.
        ar_file, _ = self._makeTransliterationData()
        en_file = os.path.join(self.get_temp_dir(), "output.txt")
        runner.infer(ar_file, predictions_file=en_file, checkpoint_path=avg_dir)
        with open(en_file) as f:
            self.assertEqual(next(f).strip(), "a t z m o n")

        # Continue the training without updating max_step
        with self.assertRaisesRegex(RuntimeError, "max_step"):
            runner.train()

    def testTrainContinueFromCheckpoint(self):
        old_model_dir = os.path.join(test_data, "transliteration-aren-v2", "v2")
        new_model_dir = os.path.join(self.get_temp_dir(), "model")
        ar_file, en_file = self._makeTransliterationData()

        config = {
            "model_dir": new_model_dir,
            "data": {
                "train_features_file": ar_file,
                "train_labels_file": en_file,
                "source_vocabulary": os.path.join(old_model_dir, "ar.vocab"),
                "target_vocabulary": os.path.join(old_model_dir, "en.vocab"),
            },
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {"batch_size": 2, "max_step": 145002},
        }

        model = load_model(old_model_dir)
        runner = Runner(model, config)
        runner.train(checkpoint_path=old_model_dir, continue_from_checkpoint=True)
        self.assertEndsWith(tf.train.latest_checkpoint(new_model_dir), "145002")

    @test_util.run_with_two_cpu_devices
    def testTrainDistribute(self):
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "data": {"train_features_file": ar_file, "train_labels_file": en_file},
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {
                "batch_size": 2,
                "length_bucket_width": None,
                "max_step": 145003,
                "single_pass": True,  # Test we do not fail when a batch is missing for a replica.
            },
        }
        runner = self._getTransliterationRunner(config)
        runner.train(num_devices=2)

    @test_util.run_with_two_cpu_devices
    def testTrainDistributeWithGradientAccumulation(self):
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "data": {"train_features_file": ar_file, "train_labels_file": en_file},
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {
                "batch_size": 2,
                "effective_batch_size": 8,
                "length_bucket_width": None,
                "max_step": 145003,
            },
        }
        runner = self._getTransliterationRunner(config)
        runner.train(num_devices=2)

    @test_util.run_with_mixed_precision
    def testTrainMixedPrecision(self):
        self.assertTrue(misc.mixed_precision_enabled())
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "data": {"train_features_file": ar_file, "train_labels_file": en_file},
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {
                "batch_size": 2,
                "length_bucket_width": None,
                "max_step": 145003,
            },
        }
        runner = self._getTransliterationRunner(config)
        runner.train()

    def testTrainWithEval(self):
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "data": {
                "train_features_file": ar_file,
                "train_labels_file": en_file,
                "eval_features_file": ar_file,
                "eval_labels_file": en_file,
            },
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {"batch_size": 10, "max_step": 145002},  # Just train for 2 steps.
            "eval": {"export_on_best": "loss"},
        }
        runner = self._getTransliterationRunner(config)
        model_dir = runner.train(with_eval=True)
        export_dir = os.path.join(model_dir, "export", "145002")
        self.assertTrue(os.path.exists(export_dir))
        self.assertTrue(tf.saved_model.contains_saved_model(export_dir))

    def testLanguageModel(self):
        src = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "src.txt"),
            ["1 2 3 4", "5 6 7 8 9", "3 2"],
        )
        vocab = test_util.make_vocab(
            os.path.join(self.get_temp_dir(), "vocab.txt"), list(map(str, range(10)))
        )
        config = {
            "model_dir": os.path.join(self.get_temp_dir(), "model"),
            "data": {
                "train_features_file": src,
                "vocabulary": vocab,
            },
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {
                "batch_size": 10,
                "max_step": 2,
            },
        }
        model = models.LanguageModel(
            decoders.SelfAttentionDecoder(2, num_units=32, ffn_inner_dim=32),
            embedding_size=16,
            reuse_embedding=False,
        )
        runner = Runner(model, config)
        runner.train()
        runner.score(src)
        runner.evaluate(src)

    def testEvaluate(self):
        ar_file, en_file = self._makeTransliterationData()
        config = {
            "params": {"beam_width": 4},
            "data": {"eval_features_file": ar_file, "eval_labels_file": en_file},
            "eval": {"external_evaluators": "BLEU"},
            "infer": {"n_best": 4},
        }
        runner = self._getTransliterationRunner(config)
        metrics = runner.evaluate()
        self.assertIn("loss", metrics)
        self.assertIn("bleu", metrics)

    @parameterized.expand([[1, "v2"], [4, "v2"], [1, "v1"]])
    def testInfer(self, beam_size, model_version):
        config = {"params": {"beam_width": beam_size}}
        runner = self._getTransliterationRunner(config, model_version)
        ar_file, _ = self._makeTransliterationData()
        en_file = os.path.join(self.get_temp_dir(), "output.txt")
        runner.infer(ar_file, predictions_file=en_file, log_time=True)
        self.assertTrue(os.path.exists(en_file))
        with open(en_file) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)
        self.assertEqual(lines[0].strip(), "a t z m o n")

    def testUpdateVocab(self):
        ar_file, en_file = self._makeTransliterationData()
        max_step = 145002
        config = {
            "data": {"train_features_file": ar_file, "train_labels_file": en_file},
            "params": {"learning_rate": 0.0005, "optimizer": "Adam"},
            "train": {"max_step": max_step, "batch_size": 10},
        }
        runner = self._getTransliterationRunner(config)

        # Reverse order of non special tokens and add a new token.
        new_en_vocab = os.path.join(self.get_temp_dir(), "en.vocab.new")
        with open(
            os.path.join(runner._config["model_dir"], "en.vocab")
        ) as en_vocab, open(new_en_vocab, "w") as new_vocab:
            tokens = en_vocab.readlines()
            for token in tokens[:3]:
                new_vocab.write(token)
            for token in reversed(tokens[3:]):
                new_vocab.write(token)
            new_vocab.write("anewtoken\n")

        output_dir = os.path.join(self.get_temp_dir(), "updated_vocab")
        self.assertEqual(
            runner.update_vocab(output_dir, tgt_vocab=new_en_vocab), output_dir
        )
        self.assertEqual(runner.model_dir, output_dir)
        self.assertTrue(
            os.path.isfile(os.path.join(output_dir, MODEL_DESCRIPTION_FILENAME))
        )

        # Check that the translation is unchanged.
        en_file = os.path.join(self.get_temp_dir(), "output.txt")
        runner.infer(ar_file, predictions_file=en_file)
        with open(en_file) as f:
            self.assertEqual(next(f).strip(), "a t z m o n")

        # We should be able to continue training without error or NaN loss.
        output_dir = runner.train()
        self.assertEndsWith(tf.train.latest_checkpoint(output_dir), str(max_step))

    def testScore(self):
        runner = self._getTransliterationRunner()
        ar_file, en_file = self._makeTransliterationData()
        score_file = os.path.join(self.get_temp_dir(), "scores.txt")
        runner.score(ar_file, en_file, output_file=score_file)
        self.assertTrue(os.path.exists(score_file))
        with open(score_file) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 5)

    @parameterized.expand([[True], [False]])
    def testExport(self, export_vocabulary_assets):
        config = {
            "data": {
                "export_vocabulary_assets": export_vocabulary_assets,
                "source_tokenization": {"mode": "char"},
            }
        }
        export_dir = os.path.join(self.get_temp_dir(), "export")
        runner = self._getTransliterationRunner(config)
        runner.export(export_dir)
        self.assertTrue(tf.saved_model.contains_saved_model(export_dir))

        # Check assets directories.
        assets = os.listdir(os.path.join(export_dir, "assets"))
        if export_vocabulary_assets:
            self.assertLen(assets, 2)
        else:
            self.assertLen(assets, 0)
        extra_assets_dir = os.path.join(export_dir, "assets.extra")
        self.assertTrue(os.path.isdir(extra_assets_dir))
        self.assertLen(os.listdir(extra_assets_dir), 1)

        # Export directory could be relocated and does not reference the original vocabulary files.
        shutil.rmtree(runner.model_dir)
        export_dir_2 = os.path.join(self.get_temp_dir(), "export_2")
        os.rename(export_dir, export_dir_2)
        self.assertTrue(tf.saved_model.contains_saved_model(export_dir_2))
        imported = tf.saved_model.load(export_dir_2)
        translate_fn = imported.signatures["serving_default"]
        outputs = translate_fn(
            tokens=tf.constant([["آ", "ت", "ز", "م", "و", "ن"]]),
            length=tf.constant([6], dtype=tf.int32),
        )
        result = tf.nest.map_structure(lambda x: x[0, 0], outputs)
        tokens = result["tokens"][: result["length"]]
        self.assertAllEqual(tokens, [b"a", b"t", b"z", b"m", b"o", b"n"])

    @parameterized.expand(
        [
            ("ctranslate2",),
            ("ctranslate2_int8",),
            ("ctranslate2_int16",),
        ]
    )
    def testCTranslate2Export(self, variant):
        try:
            import ctranslate2
        except ImportError:
            self.skipTest("ctranslate2 module is not available")
        export_dir = os.path.join(self.get_temp_dir(), "export")
        runner = self._getTransliterationRunner()
        runner.export(export_dir, exporter=exporters.make_exporter(variant))
        self.assertTrue(ctranslate2.contains_model(export_dir))
        translator = ctranslate2.Translator(export_dir)
        output = translator.translate_batch([["آ", "ت", "ز", "م", "و", "ن"]])
        self.assertListEqual(output[0][0]["tokens"], ["a", "t", "z", "m", "o", "n"])


if __name__ == "__main__":
    tf.test.main()
