import os

import numpy as np
import pytest
import tensorflow as tf

from packaging import version
from parameterized import parameterized

from opennmt import data, layers
from opennmt.models import catalog
from opennmt.tests import test_util
from opennmt.utils import exporters


def _create_vocab(temp_dir):
    vocab_path = os.path.join(temp_dir, "vocab.txt")
    vocab = test_util.make_vocab(vocab_path, ["a", "b", "c"])
    return vocab, vocab_path


def _make_model(model_template, vocab, params=None):
    model = model_template()
    model.initialize(
        dict(source_vocabulary=vocab, target_vocabulary=vocab), params=params
    )
    return model


def _create_dataset(model, temp_dir):
    data_path = os.path.join(temp_dir, "data.txt")
    test_util.make_data_file(data_path, ["a a a b b d", "a b b b", "c c"])
    dataset = model.examples_inputter.make_inference_dataset(data_path, 1)
    return dataset


def _get_predictions(model, dataset, vocab_path):
    _, tokens_to_ids, _ = data.vocab.create_lookup_tables(vocab_path)

    elem = next(iter(dataset))
    elem_ids = tf.cast(tf.squeeze(elem["ids"]), dtype=tf.dtypes.int32)

    _, pred = model(elem)
    pred_ids = tf.squeeze(tokens_to_ids.lookup(pred["tokens"]))
    tflite_concrete_fn = model.tflite_function().get_concrete_function()
    tflite_pred_ids = tflite_concrete_fn(elem_ids)

    # Model can sometimes output a tensor of shape (), if it does, converts to shape (1,)
    if tf.size(tf.shape(pred_ids)) == 0:
        pred_ids = tf.convert_to_tensor([pred_ids])

    # Modify tflite ids tensor to be the same size as normal model output
    if tf.size(pred_ids) < tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.slice(tflite_pred_ids, [0], [tf.size(pred_ids)])
    elif tf.size(pred_ids) > tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.pad(
            tflite_pred_ids, [[0, tf.size(pred_ids) - tf.size(tflite_pred_ids)]]
        )

    return pred_ids, tflite_pred_ids


def _convert_tflite(model_template, export_dir, params=None, quantization=None):
    vocab, _ = _create_vocab(export_dir)
    model = _make_model(model_template, vocab, params)
    exporter = exporters.TFLiteExporter(quantization=quantization)
    exporter.export(model, export_dir)


def dir_has_tflite_file(check_dir):
    extensions = [".lite", ".tflite"]
    for file in os.listdir(check_dir):
        for ext in extensions:
            if file.endswith(ext):
                return True
    return False


class TFLiteTest(tf.test.TestCase):
    @parameterized.expand(
        [
            [catalog.LuongAttention, {}],
            [catalog.NMTBigV1, {"replace_unknown_target": True}],
            [catalog.TransformerBase, {"beam_width": 3}],
            [catalog.TransformerRelative, {"replace_unknown_target": True}],
            [
                catalog.TransformerBaseSharedEmbeddings,
                {"replace_unknown_target": True, "beam_width": 3},
            ],
        ]
    )
    def testTFLiteOutput(self, model, params):
        vocab, vocab_path = _create_vocab(self.get_temp_dir())
        created_model = _make_model(model, vocab, params)
        dataset = _create_dataset(created_model, self.get_temp_dir())
        pred, tflite_pred = _get_predictions(created_model, dataset, vocab_path)
        self.assertAllEqual(pred, tflite_pred)

    @parameterized.expand(
        [
            [catalog.LuongAttention, {}, "float16"],
            [catalog.NMTBigV1, {"replace_unknown_target": True}, "dynamic_range"],
            [catalog.TransformerBase, {"beam_width": 3}],
            [catalog.TransformerRelative, {"replace_unknown_target": True}],
            [
                catalog.TransformerBaseSharedEmbeddings,
                {"replace_unknown_target": True, "beam_width": 3},
            ],
            [
                lambda: catalog.Transformer(
                    position_encoder_class=layers.PositionEmbedder
                )
            ],
        ]
    )
    @pytest.mark.skipif(
        version.parse(tf.__version__) >= version.parse("2.7.0"),
        reason="Test case failing with TensorFlow 2.7+",
    )
    def testTFLiteInterpreter(self, model, params=None, quantization=None):
        if params is None:
            params = {}
        export_dir = self.get_temp_dir()
        _convert_tflite(model, export_dir, params, quantization)
        self.assertTrue(dir_has_tflite_file(export_dir))
        export_file = os.path.join(export_dir, "opennmt.tflite")
        interpreter = tf.lite.Interpreter(model_path=export_file, num_threads=1)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        input_ids = [2, 3, 4, 5, 6]
        interpreter.resize_tensor_input(0, [len(input_ids)], strict=True)
        interpreter.allocate_tensors()
        np_in_data = np.array(input_ids, dtype=np.int32)
        interpreter.set_tensor(input_details[0]["index"], np_in_data)
        interpreter.invoke()
        output_ids = interpreter.get_tensor(output_details[0]["index"])
        return output_ids


if __name__ == "__main__":
    tf.test.main()
