import os
import yaml

import tensorflow as tf

from opennmt import config

config_file_1 = "config_test_1.tmp"
config_file_2 = "config_test_2.tmp"


class ConfigTest(tf.test.TestCase):

  def tearDown(self):
    if os.path.isfile(config_file_1):
      os.remove(config_file_1)
    if os.path.isfile(config_file_2):
      os.remove(config_file_2)


  def testConfigOverride(self):
    config1 = {"model_dir": "foo", "train": {"batch_size": 32, "steps": 42}}
    config2 = {"model_dir": "bar", "train": {"batch_size": 64}}

    with open(config_file_1, "w") as config_file:
      config_file.write(yaml.dump(config1))
    with open(config_file_2, "w") as config_file:
      config_file.write(yaml.dump(config2))

    loaded_config = config.load_config([config_file_1, config_file_2])

    self.assertDictEqual(
        {"model_dir": "bar", "train": {"batch_size": 64, "steps": 42}},
        loaded_config)


if __name__ == "__main__":
  tf.test.main()
