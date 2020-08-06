import os

import tensorflow as tf

from opennmt import inputters
from opennmt import models
from opennmt import training
from opennmt.tests import test_util


def _make_seq2seq_model(temp_dir):
  vocab = test_util.make_vocab(os.path.join(temp_dir, "vocab.txt"), ["1", "2", "3", "4"])
  model = models.Transformer(
      source_inputter=inputters.WordEmbedder(20),
      target_inputter=inputters.WordEmbedder(20),
      num_layers=3,
      num_units=20,
      num_heads=4,
      ffn_inner_dim=40)
  model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
  return model


class TrainingTest(tf.test.TestCase):

  def testMovingAverage(self):
    step = tf.Variable(0, trainable=False, dtype=tf.int64)
    variables = [tf.Variable(1.0), tf.Variable(2.0)]
    moving_average = training.MovingAverage(variables, step)
    variables[0].assign(3.0)
    variables[1].assign(4.0)
    moving_average.update()
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])
    with moving_average.shadow_variables():
      self.assertAllClose(self.evaluate(variables), [2.8, 3.8])
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])

  @test_util.run_with_two_cpu_devices
  def testMovingAverageDistributionStrategy(self):
    devices = tf.config.list_logical_devices(device_type="CPU")
    strategy = tf.distribute.MirroredStrategy(devices=devices)

    with strategy.scope():
      variables = [tf.Variable(1.0), tf.Variable(2.0)]
      step = tf.Variable(0, trainable=False, dtype=tf.int64)

    moving_average = training.MovingAverage(variables, step)
    variables[0].assign(3.0)
    variables[1].assign(4.0)
    moving_average.update()
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])
    with moving_average.shadow_variables():
      self.assertAllClose(self.evaluate(variables), [2.8, 3.8])
    self.assertAllEqual(self.evaluate(variables), [3.0, 4.0])

  def testEmptyTrainingDataset(self):
    model = _make_seq2seq_model(self.get_temp_dir())
    optimizer = tf.keras.optimizers.SGD(1.0)
    trainer = training.Trainer(model, optimizer)

    empty_file = os.path.join(self.get_temp_dir(), "train.txt")
    with open(empty_file, "w"):
      pass
    dataset = model.examples_inputter.make_training_dataset(empty_file, empty_file, 32)

    with self.assertRaisesRegex(RuntimeError, "No training steps"):
      trainer(dataset)


if __name__ == "__main__":
  tf.test.main()
