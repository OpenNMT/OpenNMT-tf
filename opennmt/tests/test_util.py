import os
import unittest

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops

from opennmt import constants
from opennmt.data import vocab
from opennmt.utils import compat, misc


def skip_if_unsupported(symbol):
    return unittest.skipIf(not compat.tf_supports(symbol), "tf.%s is not supported")


def get_test_data_dir():
    test_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.join(test_dir, "..", "..")
    test_data = os.path.join(root_dir, "testdata")
    return test_data


def make_data_file(path, lines):
    with open(path, "w") as data:
        for line in lines:
            data.write("%s\n" % line)
    return path


def make_vocab(path, tokens):
    vocabulary = vocab.Vocab(
        special_tokens=[
            constants.PADDING_TOKEN,
            constants.START_OF_SENTENCE_TOKEN,
            constants.END_OF_SENTENCE_TOKEN,
        ]
    )
    for token in tokens:
        vocabulary.add(token)
    vocabulary.serialize(path)
    return path


def make_vocab_from_file(path, data_file):
    vocabulary = vocab.Vocab(
        special_tokens=[
            constants.PADDING_TOKEN,
            constants.START_OF_SENTENCE_TOKEN,
            constants.END_OF_SENTENCE_TOKEN,
        ]
    )
    vocabulary.add_from_text(data_file)
    vocabulary.serialize(path)
    return path


def _reset_context():
    # See github.com/tensorflow/tensorflow/blob/master/tensorflow/python/framework/config_test.py
    # TODO: find a way to achieve that without relying on TensorFlow private APIs.
    context._context = None
    ops.enable_eager_execution_internal()


def run_with_two_cpu_devices(fn):
    """Defines 2 logical devices before running :obj:`fn`."""

    def decorator(*args, **kwargs):
        _reset_context()
        physical_devices = tf.config.list_physical_devices("CPU")
        if len(physical_devices) == 1:
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [
                    tf.config.LogicalDeviceConfiguration(),
                    tf.config.LogicalDeviceConfiguration(),
                ],
            )
        try:
            return fn(*args, **kwargs)
        finally:
            _reset_context()

    return decorator


def run_with_mixed_precision(fn):
    """Enables mixed precision before running :obj:`fn`."""

    def decorator(*args, **kwargs):
        misc.enable_mixed_precision(force=True)
        try:
            return fn(*args, **kwargs)
        finally:
            misc.disable_mixed_precision()

    return decorator
