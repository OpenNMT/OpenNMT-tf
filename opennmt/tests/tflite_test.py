import os

import tensorflow as tf

from opennmt import data
from opennmt.tests import test_util
from opennmt.models import catalog
from opennmt.utils import exporters
from parameterized import parameterized


def _create_vocab(temp_dir):
    vocab_path = os.path.join(temp_dir, "vocab.txt")
    vocab = test_util.make_vocab(vocab_path, ["a", "b", "c"])
    return vocab, vocab_path


def _make_model(model_template, vocab):
    model = model_template()
    model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
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

    # Modify tflite ids tensor to be the same size as normal model output
    if tf.size(pred_ids) < tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.slice(tflite_pred_ids, [0], [tf.size(pred_ids)])
    elif tf.size(pred_ids) > tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.pad(
            tflite_pred_ids, [[tf.size(pred_ids) - tf.size(tflite_pred_ids)]]
        )

    return pred_ids, tflite_pred_ids


def _convert_tflite(model_template, export_dir):
    vocab, _ = _create_vocab(export_dir)
    model = _make_model(model_template, vocab)
    exporter = exporters.TFLiteExporter()
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
            [catalog.LuongAttention],
            [catalog.NMTBigV1],
        ]
    )
    def testTFLiteOutput(self, model):
        vocab, vocab_path = _create_vocab(self.get_temp_dir())
        created_model = _make_model(model, vocab)
        dataset = _create_dataset(created_model, self.get_temp_dir())
        pred, tflite_pred = _get_predictions(created_model, dataset, vocab_path)
        self.assertAllEqual(pred, tflite_pred)

    @parameterized.expand(
        [
            [catalog.LuongAttention],
            [catalog.NMTBigV1],
        ]
    )
    def testTFLiteOutputFile(self, model):
        export_dir = self.get_temp_dir()
        _convert_tflite(model, export_dir)
        self.assertTrue(dir_has_tflite_file(export_dir))


if __name__ == "__main__":
    tf.test.main()
