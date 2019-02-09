"""Base class for encoders and generic multi encoders."""

import abc
import six

import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class Encoder(tf.keras.layers.Layer):
  """Base class for encoders."""
  pass
