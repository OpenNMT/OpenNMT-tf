import itertools

import numpy as np
import tensorflow as tf

from parameterized import parameterized

from opennmt import models
from opennmt.utils import misc


class MiscTest(tf.test.TestCase):
    @parameterized.expand(itertools.product((True, False), repeat=2))
    def testGetVariableName(self, distributed_variables, mixed_precision):
        if mixed_precision:
            misc.enable_mixed_precision(force=True)
            self.assertTrue(misc.mixed_precision_enabled())

        if distributed_variables:
            devices = tf.config.list_logical_devices(device_type="CPU")
            strategy = tf.distribute.MirroredStrategy(devices=devices)
        else:
            strategy = None

        class Layer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.variable = self.add_weight("variable", [42])

        class Model(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.layers = [Layer()]

        if strategy is not None:
            with strategy.scope():
                model = Model()
        else:
            model = Model()

        variable = model.layers[0].variable
        expected_name = "model/layers/0/variable/.ATTRIBUTES/VARIABLE_VALUE"
        variable_name = misc.get_variable_name(variable, model)
        self.assertEqual(variable_name, expected_name)

        variables = misc.get_variables_name_mapping(model, "model")
        self.assertIs(variables[expected_name], variable)

        if mixed_precision:
            misc.disable_mixed_precision()
            self.assertFalse(misc.mixed_precision_enabled())

    def testSetDropout(self):
        class Layer(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.dropout = 0.3
                self.x = tf.keras.layers.Dropout(0.2)

        class Model(tf.keras.layers.Layer):
            def __init__(self):
                super().__init__()
                self.output_dropout = 0.1
                self.layer = Layer()
                self.layers = [Layer(), Layer()]

        model = Model()
        misc.set_dropout(model, 0.5)
        self.assertEqual(model.output_dropout, 0.5)
        self.assertEqual(model.layer.dropout, 0.5)
        self.assertEqual(model.layer.x.rate, 0.5)
        self.assertEqual(model.layers[1].dropout, 0.5)
        self.assertEqual(model.layers[1].x.rate, 0.5)

    def testFormatTranslationOutput(self):
        self.assertEqual(misc.format_translation_output("hello world"), "hello world")
        self.assertEqual(
            misc.format_translation_output("hello world", score=42),
            "%f ||| hello world" % 42,
        )
        self.assertEqual(
            misc.format_translation_output(
                "hello world", score=42, token_level_scores=[24, 64]
            ),
            "%f ||| hello world ||| %f %f" % (42, 24, 64),
        )
        self.assertEqual(
            misc.format_translation_output("hello world", token_level_scores=[24, 64]),
            "hello world ||| %f %f" % (24, 64),
        )
        self.assertEqual(
            misc.format_translation_output(
                "hello world", attention=[[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]
            ),
            "hello world",
        )
        self.assertEqual(
            misc.format_translation_output(
                "hello world",
                attention=np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]),
                alignment_type="hard",
            ),
            "hello world ||| 1-0 0-1",
        )
        self.assertEqual(
            misc.format_translation_output(
                "hello world",
                attention=np.array([[0.1, 0.7, 0.2], [0.5, 0.3, 0.2]]),
                alignment_type="soft",
            ),
            "hello world ||| 0.100000 0.700000 0.200000 ; 0.500000 0.300000 0.200000",
        )

    def testReadSummaries(self):
        event_dir = self.get_temp_dir()
        summary_writer = tf.summary.create_file_writer(event_dir)
        with summary_writer.as_default():
            tf.summary.scalar("values/a", 1, step=0)
            tf.summary.scalar("values/b", 2, step=0)
            tf.summary.scalar("values/a", 3, step=5)
            tf.summary.scalar("values/b", 4, step=5)
            tf.summary.scalar("values/a", 5, step=10)
            tf.summary.scalar("values/b", 6, step=10)
            summary_writer.flush()
        summaries = misc.read_summaries(event_dir)
        self.assertLen(summaries, 3)
        steps, values = zip(*summaries)
        self.assertListEqual(list(steps), [0, 5, 10])
        values = list(values)
        self.assertDictEqual(values[0], {"values/a": 1, "values/b": 2})
        self.assertDictEqual(values[1], {"values/a": 3, "values/b": 4})
        self.assertDictEqual(values[2], {"values/a": 5, "values/b": 6})

    def testEventOrderRestorer(self):
        events = []
        restorer = misc.OrderRestorer(
            index_fn=lambda x: x[0], callback_fn=lambda x: events.append(x)
        )
        self.assertFalse(restorer.push((2, "toto")))
        self.assertFalse(restorer.push((1, "tata")))
        self.assertFalse(restorer.push((3, "foo")))
        self.assertTrue(restorer.push((0, "bar")))
        self.assertTrue(restorer.push((4, "titi")))
        with self.assertRaises(ValueError):
            restorer.push((2, "invalid"))
        self.assertEqual(len(events), 5)
        self.assertTupleEqual(events[0], (0, "bar"))
        self.assertTupleEqual(events[1], (1, "tata"))
        self.assertTupleEqual(events[2], (2, "toto"))
        self.assertTupleEqual(events[3], (3, "foo"))
        self.assertTupleEqual(events[4], (4, "titi"))

    def testClassRegistry(self):
        registry = misc.ClassRegistry(base_class=models.Model)
        self.assertIsNone(registry.get("TransformerBig"))
        registry.register(models.TransformerBig)
        self.assertEqual(registry.get("TransformerBig"), models.TransformerBig)
        registry.register(models.TransformerBig, name="TransformerLarge")
        self.assertEqual(registry.get("TransformerLarge"), models.TransformerBig)
        self.assertSetEqual(
            registry.class_names, set(["TransformerBig", "TransformerLarge"])
        )

        registry.register(models.TransformerBaseRelative, alias="TransformerRelative")
        self.assertEqual(
            registry.get("TransformerBaseRelative"), models.TransformerBaseRelative
        )
        self.assertEqual(
            registry.get("TransformerRelative"), models.TransformerBaseRelative
        )

        with self.assertRaises(ValueError):
            registry.register(models.TransformerBig)
        with self.assertRaises(TypeError):
            registry.register(misc.ClassRegistry)

    def testRelativeConfig(self):
        config = misc.RelativeConfig({"a_1": 1, "1": 2, "2": 3}, prefix="a_")
        self.assertEqual(config["1"], 1)
        self.assertEqual(config["2"], 3)
        with self.assertRaisesRegex(KeyError, "a_3"):
            _ = config["3"]


if __name__ == "__main__":
    tf.test.main()
