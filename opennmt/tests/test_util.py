import unittest

from opennmt.utils import compat


def run_tf1_only(func):
  return unittest.skipIf(compat.is_tf2(), "TensorFlow v1 only test")(func)

def run_tf2_only(func):
  return unittest.skipIf(not compat.is_tf2(), "TensorFlow v2 only test")(func)
