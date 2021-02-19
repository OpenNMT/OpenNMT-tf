import os

import tensorflow as tf

from opennmt import models
from opennmt import data
from opennmt.tests import test_util


def _create_vocab(temp_dir):
    vocab_path = os.path.join(temp_dir, "vocab.txt")
    vocab = test_util.make_vocab(vocab_path, ["a", "b", "c"])
    return vocab, vocab_path


def _make_model(model_template, vocab):
    model = model_template()
    model.initialize(dict(source_vocabulary=vocab, target_vocabulary=vocab))
    model.create_variables()
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
    tflite_concrete_fn = tf.function(
        model.infer_tflite,
        input_signature=[tf.TensorSpec([None], dtype=tf.dtypes.int32, name="ids")],
    ).get_concrete_function()
    tflite_pred_ids = tflite_concrete_fn(elem_ids)

    # Modify tflite ids tensor to be the same size as normal model output
    if tf.size(pred_ids) < tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.slice(tflite_pred_ids, [0], [tf.size(pred_ids)])
    elif tf.size(pred_ids) > tf.size(tflite_pred_ids):
        tflite_pred_ids = tf.pad(
            tflite_pred_ids, [[tf.size(pred_ids) - tf.size(tflite_pred_ids)]]
        )

    return pred_ids, tflite_pred_ids


def _convert_tflite(model_template, tflite_model_path, temp_dir):
    vocab, _ = _create_vocab(temp_dir)
    model = _make_model(model_template, vocab)
    tflite_concrete_fn = tf.function(
        model.infer_tflite,
        input_signature=[tf.TensorSpec([None], dtype=tf.dtypes.int32, name="ids")],
    ).get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([tflite_concrete_fn])
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with tf.io.gfile.GFile(tflite_model_path, "wb") as f:
        f.write(tflite_model)


class TFLiteTest(tf.test.TestCase):
    def testLuongAttentionTFLiteOutput(self):
        vocab, vocab_path = _create_vocab(self.get_temp_dir())
        model = _make_model(models.LuongAttention, vocab)
        dataset = _create_dataset(model, self.get_temp_dir())
        pred, tflite_pred = _get_predictions(model, dataset, vocab_path)
        self.assertAllEqual(pred, tflite_pred)

    def testNMTBigV1TFLiteOutput(self):
        vocab, vocab_path = _create_vocab(self.get_temp_dir())
        model = _make_model(models.NMTBigV1, vocab)
        dataset = _create_dataset(model, self.get_temp_dir())
        pred, tflite_pred = _get_predictions(model, dataset, vocab_path)
        self.assertAllEqual(pred, tflite_pred)

    def testLuongAttentionTFLiteFile(self):
        tflite_model_path = os.path.join(self.get_temp_dir(), "opennmt.tflite")
        _convert_tflite(models.LuongAttention, tflite_model_path, self.get_temp_dir())
        self.assertTrue(os.path.isfile(tflite_model_path))

    def testNMTBigTFLiteFile(self):
        tflite_model_path = os.path.join(self.get_temp_dir(), "opennmt.tflite")
        _convert_tflite(models.NMTBigV1, tflite_model_path, self.get_temp_dir())
        self.assertTrue(os.path.isfile(tflite_model_path))


if __name__ == "__main__":
    tf.test.main()
