import os

import tensorflow as tf

from opennmt.utils import checkpoint as checkpoint_util


class _CustomDense(tf.keras.layers.Dense):
    def add_weight(self, name, *args, **kwargs):
        # This is to test the case where the variable name is different than the attribute name.
        name += "_1"
        return super().add_weight(name, *args, **kwargs)


class _DummyModel(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.layers = [tf.keras.layers.Dense(20), _CustomDense(20)]

    def call(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CheckpointTest(tf.test.TestCase):
    def testLastSavedStep(self):
        model = _DummyModel()
        model(tf.random.uniform([4, 10]))
        model_dir = os.path.join(self.get_temp_dir(), "model")
        checkpoint = checkpoint_util.Checkpoint(model, model_dir=model_dir)
        self.assertIsNone(checkpoint.last_saved_step)
        checkpoint.save(10)
        self.assertEqual(checkpoint.last_saved_step, 10)
        checkpoint.save(20)
        self.assertEqual(checkpoint.last_saved_step, 20)

        # Property should not be bound to an instance.
        checkpoint = checkpoint_util.Checkpoint(model, model_dir=model_dir)
        self.assertEqual(checkpoint.last_saved_step, 20)

    def testCheckpointAveraging(self):
        model = _DummyModel()
        optimizer = tf.keras.optimizers.Adam()

        @tf.function
        def _build_model():
            x = tf.random.uniform([4, 10])
            y = model(x)
            loss = tf.reduce_mean(y)
            gradients = optimizer.get_gradients(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        def _assign_var(var, scalar):
            var.assign(tf.ones_like(var) * scalar)

        def _all_equal(var, scalar):
            return tf.size(tf.where(tf.not_equal(var, scalar))).numpy() == 0

        def _get_var_list(checkpoint_path):
            return [name for name, _ in tf.train.list_variables(checkpoint_path)]

        _build_model()

        # Write some checkpoint with all variables set to the step value.
        steps = [10, 20, 30, 40]
        num_checkpoints = len(steps)
        avg_value = sum(steps) / num_checkpoints
        directory = os.path.join(self.get_temp_dir(), "src")
        checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint, directory, max_to_keep=num_checkpoints
        )
        for step in steps:
            _assign_var(model.layers[0].kernel, step)
            _assign_var(model.layers[0].bias, step)
            checkpoint_manager.save(checkpoint_number=step)

        output_dir = os.path.join(self.get_temp_dir(), "dst")
        checkpoint_util.average_checkpoints(
            directory, output_dir, dict(model=model, optimizer=optimizer)
        )
        avg_checkpoint = tf.train.latest_checkpoint(output_dir)
        self.assertIsNotNone(avg_checkpoint)
        self.assertEqual(
            checkpoint_util.get_step_from_checkpoint_prefix(avg_checkpoint), steps[-1]
        )
        checkpoint.restore(avg_checkpoint)
        self.assertTrue(_all_equal(model.layers[0].kernel, avg_value))
        self.assertTrue(_all_equal(model.layers[0].bias, avg_value))
        self.assertListEqual(
            _get_var_list(avg_checkpoint),
            _get_var_list(checkpoint_manager.latest_checkpoint),
        )


if __name__ == "__main__":
    tf.test.main()
