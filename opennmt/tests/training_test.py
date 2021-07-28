import json
import os

import tensorflow as tf

from opennmt import inputters, models, training
from opennmt.tests import test_util


def _make_seq2seq_model(temp_dir):
    vocab = test_util.make_vocab(
        os.path.join(temp_dir, "vocab.txt"), ["1", "2", "3", "4"]
    )
    model = models.Transformer(
        source_inputter=inputters.WordEmbedder(20),
        target_inputter=inputters.WordEmbedder(20),
        num_layers=3,
        num_units=20,
        num_heads=4,
        ffn_inner_dim=40,
    )
    model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
    return model


class TrainingTest(tf.test.TestCase):
    def testMovingAverage(self):
        step = tf.Variable(0, trainable=False, dtype=tf.int64)
        variables = [tf.Variable(1.0), tf.Variable(2.0)]
        moving_average = training.MovingAverage(variables, step)
        moving_average.update()
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
        with strategy.scope():
            moving_average.update()

        variables[0].assign(3.0)
        variables[1].assign(4.0)

        with strategy.scope():
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
        dataset = model.examples_inputter.make_training_dataset(
            empty_file, empty_file, 32
        )

        with self.assertRaisesRegex(RuntimeError, "No training steps"):
            trainer(dataset)

    def testTrainingStats(self):
        model = _make_seq2seq_model(self.get_temp_dir())
        optimizer = tf.keras.optimizers.SGD(1.0)
        stats = training.TrainingStats(model, optimizer, warmup_steps=2)

        def _generate_example(length):
            return tf.constant(" ".join(map(str, range(length))))

        def _step(source_length, target_length, step, loss):
            source_example = _generate_example(source_length)
            target_example = _generate_example(target_length)
            source_features = model.features_inputter.make_features(source_example)
            target_features = model.labels_inputter.make_features(target_example)
            stats.update_on_example(source_features, target_features)
            stats.update_on_step(tf.constant(step, dtype=tf.int64), tf.constant(loss))

        def _is_json_serializable(summary):
            json.dumps(summary)
            return True

        _step(24, 23, 5, 9.8)
        _step(10, 8, 10, 9.6)

        summary = stats.get_last_summary()
        self.assertTrue(_is_json_serializable(summary))
        self.assertEqual(summary["learning_rate"], 1.0)
        self.assertEqual(summary["step"], 10)
        self.assertAllClose(summary["loss"], 9.6)

        # Throughput values are ignored in the 2 first steps.
        self.assertEqual(summary["steps_per_sec"], 0)
        self.assertEqual(summary["words_per_sec"]["source"], 0)
        self.assertEqual(summary["words_per_sec"]["target"], 0)

        _step(14, 21, 15, 9.4)

        summary = stats.get_last_summary()
        self.assertNotEqual(summary["steps_per_sec"], 0)
        self.assertNotEqual(summary["words_per_sec"]["source"], 0)
        self.assertNotEqual(summary["words_per_sec"]["target"], 0)

        stats.log()
        stats.reset_throughput()

        summary = stats.get_last_summary()
        self.assertEqual(summary["steps_per_sec"], 0)
        self.assertEqual(summary["words_per_sec"]["source"], 0)
        self.assertEqual(summary["words_per_sec"]["target"], 0)

        summary = stats.get_global_summary()
        self.assertTrue(_is_json_serializable(summary))
        self.assertEqual(summary["last_learning_rate"], 1.0)
        self.assertEqual(summary["last_step"], 15)
        self.assertAllClose(summary["last_loss"], 9.4)
        self.assertAllClose(summary["average_loss"], 9.6)
        self.assertEqual(summary["num_steps"], 3)


if __name__ == "__main__":
    tf.test.main()
