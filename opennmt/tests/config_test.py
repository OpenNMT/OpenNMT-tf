import os
import yaml

import tensorflow as tf

from opennmt import config
from opennmt.models.model import Model


class ConfigTest(tf.test.TestCase):

  def testConfigOverride(self):
    config1 = {"model_dir": "foo", "train": {"batch_size": 32, "steps": 42}}
    config2 = {"model_dir": "bar", "train": {"batch_size": 64}}
    config_file_1 = os.path.join(self.get_temp_dir(), "config1.yml")
    config_file_2 = os.path.join(self.get_temp_dir(), "config2.yml")

    with open(config_file_1, mode="wb") as config_file:
      config_file.write(tf.compat.as_bytes(yaml.dump(config1)))
    with open(config_file_2, mode="wb") as config_file:
      config_file.write(tf.compat.as_bytes(yaml.dump(config2)))

    loaded_config = config.load_config([config_file_1, config_file_2])

    self.assertDictEqual(
        {"model_dir": "bar", "train": {"batch_size": 64, "steps": 42}},
        loaded_config)

  def testLoadModelModule(self):
    model_path = os.path.join(self.get_temp_dir(), "model.py")
    with open(model_path, mode="wb") as model_file:
      model_file.write(b"model = lambda: 42")
    model_module = config.load_model_module(model_path)
    model = model_module.model()
    self.assertEqual(42, model)

  def testLoadModelFromCatalog(self):
    model_name = "NMTSmall"
    model = config.load_model_from_catalog(model_name)
    self.assertIsInstance(model, Model)

  def testLoadModel(self):
    model_name = "NMTSmall"
    model_dir = self.get_temp_dir()
    model = config.load_model(model_dir, model_name=model_name)
    self.assertIsInstance(model, Model)
    model = config.load_model(model_dir)
    self.assertIsInstance(model, Model)

  def testLoadModelInvalidArguments(self):
    with self.assertRaises(ValueError):
      config.load_model(self.get_temp_dir(), model_file="a", model_name="b")

  def testLoadModelMissingModel(self):
    with self.assertRaises(RuntimeError):
      config.load_model(self.get_temp_dir())


if __name__ == "__main__":
  tf.test.main()
