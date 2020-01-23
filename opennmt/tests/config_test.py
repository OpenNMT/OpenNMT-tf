import copy
import os
import filecmp
import yaml

from parameterized import parameterized

import tensorflow as tf

from opennmt import config
from opennmt.models.model import Model


class ConfigTest(tf.test.TestCase):

  def testConfigOverride(self):
    config1 = {"model_dir": "foo", "train": {"batch_size": 32, "steps": 42}}
    config2 = {"model_dir": "bar", "train": {"batch_size": 64}}
    config_file_1 = os.path.join(self.get_temp_dir(), "config1.yml")
    config_file_2 = os.path.join(self.get_temp_dir(), "config2.yml")

    with open(config_file_1, mode="w") as config_file:
      config_file.write(yaml.dump(config1))
    with open(config_file_2, mode="w") as config_file:
      config_file.write(yaml.dump(config2))

    loaded_config = config.load_config([config_file_1, config_file_2])

    self.assertDictEqual(
        {"model_dir": "bar", "train": {"batch_size": 64, "steps": 42}},
        loaded_config)

  def _writeCustomModel(self, filename="test_model.py", return_value=42):
    model_path = os.path.join(self.get_temp_dir(), filename)
    with open(model_path, mode="w") as model_file:
      model_file.write("model = lambda: %d" % return_value)
    return model_path

  def testLoadModelModule(self):
    model_path = self._writeCustomModel()
    model_module = config.load_model_module(model_path)
    model = model_module.model()
    self.assertEqual(42, model)

  def testLoadModelFromCatalog(self):
    model_name = "Transformer"
    model = config.load_model_from_catalog(model_name)
    self.assertIsInstance(model, Model)

  @parameterized.expand([
      ("Transformer",),
      ("TransformerBase",),
  ])
  def testLoadModel(self, model_name):
    model_dir = self.get_temp_dir()
    model = config.load_model(model_dir, model_name=model_name)
    self.assertTrue(os.path.exists(os.path.join(model_dir, "model_description.py")))
    self.assertIsInstance(model, Model)
    model = config.load_model(model_dir)
    self.assertIsInstance(model, Model)

  def testLoadModelDescriptionCompat(self):
    model_dir = self.get_temp_dir()
    description = os.path.join(model_dir, "model_description.py")
    with open(description, "w") as description_file:
      description_file.write("from opennmt.models import catalog\n")
      description_file.write("model = catalog.Transformer\n")
    model = config.load_model(model_dir)
    self.assertIsInstance(model, Model)

  def testLoadModelFile(self):
    model_file = self._writeCustomModel()
    model_dir = self.get_temp_dir()
    model = config.load_model(model_dir, model_file=model_file)
    saved_description_path = os.path.join(model_dir, "model_description.py")
    self.assertTrue(os.path.exists(saved_description_path))
    self.assertTrue(filecmp.cmp(model_file, saved_description_path))
    self.assertEqual(model, 42)
    model = config.load_model(model_dir)
    self.assertEqual(model, 42)

  def testLoadModelFileOverride(self):
    model_dir = self.get_temp_dir()
    saved_description_path = os.path.join(model_dir, "model_description.py")
    model_file = self._writeCustomModel(filename="test_model1.py", return_value=1)
    model = config.load_model(model_dir, model_file=model_file)
    self.assertTrue(filecmp.cmp(model_file, saved_description_path))
    model_file = self._writeCustomModel(filename="test_model2.py", return_value=2)
    model = config.load_model(model_dir, model_file=model_file)
    self.assertTrue(filecmp.cmp(model_file, saved_description_path))

  def testLoadModelInvalidArguments(self):
    with self.assertRaises(ValueError):
      config.load_model(self.get_temp_dir(), model_file="a", model_name="b")

  def testLoadModelInvalidInvalidName(self):
    with self.assertRaisesRegex(ValueError, "does not exist"):
      config.load_model(self.get_temp_dir(), model_name="b")

  def testLoadModelInvalidInvalidFile(self):
    with self.assertRaisesRegex(ValueError, "not found"):
      config.load_model(self.get_temp_dir(), model_file="a")

  def testLoadModelMissingModel(self):
    with self.assertRaises(RuntimeError):
      config.load_model(self.get_temp_dir())

  def testConvertToV2Config(self):
    v1_config = {
        "data": {
            "source_words_vocabulary": "a.txt",
            "target_words_vocabulary": "b.txt",
        },
        "params": {
            "optimizer": "LazyAdamOptimizer",
            "optimizer_params": {
                "beta1": 0.9,
                "beta2": 0.998,
            },
            "param_init": 0.1,
            "loss_scale": 2,
            "horovod": {},
            "maximum_learning_rate": 4,
            "maximum_iterations": 250,
            "clip_gradients": 5,
            "weight_decay": 0.1,
            "decay_step_duration": 8,
            "gradients_accum": 9,
            "decay_type": "noam_decay",
            "decay_rate": 512,
            "decay_steps": 4000
        },
        "train": {
            "batch_size": 64,
            "num_threads": 4,
            "prefetch_buffer_size": 1,
            "bucket_width": 1,
            "train_steps": 500000,
            "save_checkpoints_secs": 600,
        },
        "eval": {
            "batch_size": 32,
            "eval_delay": 3600,
            "exporters": "best",
            "num_threads": 4,
            "prefetch_buffer_size": 1,
        },
        "infer": {
            "batch_size": 32,
            "num_threads": 4,
            "prefetch_buffer_size": 1,
            "bucket_width": 1,
        },
        "score": {
            "num_threads": 4,
            "prefetch_buffer_size": 1,
        },
    }

    expected_v2_config = {
        "data": {
            "source_vocabulary": "a.txt",
            "target_vocabulary": "b.txt",
        },
        "params": {
            "optimizer": "LazyAdam",
            "optimizer_params": {
                "beta_1": 0.9,
                "beta_2": 0.998,
                "clipnorm": 5,
                "weight_decay": 0.1
            },
            "decay_type": "NoamDecay",
            "decay_params": {
                "model_dim": 512,
                "warmup_steps": 4000
            },
            "maximum_decoding_length": 250,
        },
        "train": {
            "batch_size": 64,
            "effective_batch_size": 64 * max(8, 9),
            "length_bucket_width": 1,
            "max_step": 500000,
        },
        "eval": {
            "batch_size": 32,
        },
        "infer": {
            "batch_size": 32,
            "length_bucket_width": 1,
        },
    }

    original_v1_config = copy.deepcopy(v1_config)
    v2_config = config.convert_to_v2_config(v1_config)
    self.assertDictEqual(v2_config, expected_v2_config)
    self.assertDictEqual(v1_config, original_v1_config)
    self.assertDictEqual(config.convert_to_v2_config(v2_config), expected_v2_config)


if __name__ == "__main__":
  tf.test.main()
