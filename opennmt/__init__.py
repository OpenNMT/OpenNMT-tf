"""OpenNMT module."""

import tensorflow as tf


__version__ = "1.19.0"

if tf.__version__.startswith("2"):
  from opennmt.v2 import *  # pylint: disable=wildcard-import
else:
  from opennmt import decoders
  from opennmt import encoders
  from opennmt import inputters
  from opennmt import layers
  from opennmt import models
  from opennmt import tokenizers

  from opennmt.runner import Runner
