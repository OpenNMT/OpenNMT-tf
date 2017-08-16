"""Template file from model configurations."""

# You usually want to import tensorflow and opennmt modules.
import tensorflow as tf
import opennmt as onmt

def model():
  """Builds a model.

  Returns:
    A `opennmt.models.Model`.
  """
  pass

def train(model):
  """Run training specific code.

  You usually call methods on `model` to set training specific
  attributes, e.g. the longest sequence lengths accepted.

  Args:
    model: The model previously built.
  """
  pass

def infer(model):
  """Run inference specific code.

  Similar to `train` but for inference.

  Args:
    model: The model previously built.
  """
  pass
