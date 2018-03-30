import io
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

    with io.open(config_file_1, encoding="utf-8", mode="w") as config_file:
      try:
        config_file.write(yaml.dump(config1))
      except TypeError:
        config_file.write(unicode(yaml.dump(config1)))
    with io.open(config_file_2, encoding="utf-8", mode="w") as config_file:
      try:
        config_file.write(yaml.dump(config2))
      except TypeError:
        config_file.write(unicode(yaml.dump(config2)))

    loaded_config = config.load_config([config_file_1, config_file_2])

    self.assertDictEqual(
        {"model_dir": "bar", "train": {"batch_size": 64, "steps": 42}},
        loaded_config)


if __name__ == "__main__":
  tf.test.main()
