import os
import yaml

import tensorflow as tf

from opennmt import config


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


if __name__ == "__main__":
  tf.test.main()
