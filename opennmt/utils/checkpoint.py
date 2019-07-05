"""Checkpoint utilities."""

import os
import six

import tensorflow as tf
import numpy as np

from opennmt.utils import misc


class Checkpoint(object):
  """Wrapper around TensorFlow checkpoints utilities."""

  def __init__(self, model, optimizer=None, model_dir=None, keep_checkpoint_max=8):
    """Initializes the wrapper.

    Args:
      model: A :class:`opennmt.models.model.Model` to save.
      optimizer: The optimizer instance.
      model_dir: The directory where checkpoints will be saved. If not set, a
        temporary directory will be used.
      keep_checkpoint_max: The maximum number of checkpoints to keep.
    """
    if model_dir is None:
      model_dir = tempfile.mkdtemp()
    trackables = {}
    trackables["model"] = model
    if optimizer is not None:
      trackables["optimizer"] = optimizer
    self._model = model
    self._optimizer = optimizer
    self._model_dir = model_dir
    self._checkpoint = tf.train.Checkpoint(**trackables)
    self._checkpoint_manager = tf.train.CheckpointManager(
        self._checkpoint, model_dir, keep_checkpoint_max)

  @property
  def model(self):
    """The managed model."""
    return self._model

  @property
  def optimizer(self):
    """The managed optimizer."""
    return self._optimizer

  @property
  def model_dir(self):
    """The model directory."""
    return self._model_dir

  def save(self, step):
    """Saves a checkpoint for :obj:`step`."""
    path = self._checkpoint_manager.save(checkpoint_number=step)
    tf.get_logger().info("Saved checkpoint %s", path)

  def restore(self, checkpoint_path=None, weights_only=False):
    """Restores a checkpoint:

    Args:
      checkpoint_path: Path a checkpoint to restore. If not set, the latest
        checkpoint from :obj:`model_dir` will be restored.
      weights_only: Only restore model weights.

    Returns:
      Path to the restored checkpoint.
    """
    if weights_only:
      checkpoint = tf.train.Checkpoint(model=self._model)
    else:
      checkpoint = self._checkpoint
    if checkpoint_path is not None:
      if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    elif self._checkpoint_manager.latest_checkpoint is not None:
      checkpoint_path = self._checkpoint_manager.latest_checkpoint
    if checkpoint_path is None:
      tf.get_logger().warning("No checkpoint to restore in %s", self._model_dir)
      return None
    is_v1 = os.path.basename(checkpoint_path).startswith("model")
    if is_v1:
      _restore_v1_checkpoint(
          checkpoint_path,
          self._model,
          optimizer=self._optimizer if not weights_only else None)
    else:
      checkpoint.restore(checkpoint_path)
    tf.get_logger().info("Restored checkpoint %s", checkpoint_path)
    return checkpoint_path


def _restore_v1_checkpoint(checkpoint_path, model, optimizer=None):
  raise NotImplementedError("V1 checkpoints are currently unsupported")

def get_checkpoint_variables(checkpoint_path):
  """Returns variables included in a checkpoint.

  Args:
    checkpoint_path: Path to the checkpoint.

  Returns:
    A dictionary mapping variables name to value.
  """
  reader = tf.train.load_checkpoint(checkpoint_path)
  return {
      name:reader.get_tensor(name)
      for name in six.iterkeys(reader.get_variable_to_shape_map())}

def average_checkpoints(model_dir,
                        output_dir,
                        trackables,
                        max_count=8,
                        model_key="model"):
  """Averages object-based checkpoints.

  Args:
    model_dir: The directory containing checkpoints.
    output_dir: The directory that will contain the averaged checkpoint.
    trackables: A dictionary containing the trackable objects included in the
      checkpoint.
    max_count: The maximum number of checkpoints to average.
    model_key: The key in :obj:`trackables` that references the model.

  Returns:
    The path to the directory containing the averaged checkpoint.

  Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
    ValueError: if a model is not found in :obj:`trackables` or is not already
      built.
    ValueError: if no checkpoints are found in :obj:`model_dir`.
  """
  if model_dir == output_dir:
    raise ValueError("Model and output directory must be different")
  model = trackables.get(model_key)
  if model is None:
    raise ValueError("%s not found in trackables %s" % (model_key, trackables))
  if not model.built:
    raise ValueError("The model should be built before calling this function")

  checkpoint = tf.train.Checkpoint(**trackables)
  checkpoint_manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=None)

  checkpoints_path = checkpoint_manager.checkpoints
  if not checkpoints_path:
    raise ValueError("No checkpoints found in %s" % model_dir)
  if len(checkpoints_path) > max_count:
    checkpoints_path = checkpoints_path[-max_count:]
  num_checkpoints = len(checkpoints_path)
  last_step = int(checkpoints_path[-1].split("-")[-1])

  tf.get_logger().info("Averaging %d checkpoints...", num_checkpoints)
  for i, checkpoint_path in enumerate(reversed(checkpoints_path)):
    tf.get_logger().info("Reading checkpoint %s...", checkpoint_path)
    if i == 0:
      checkpoint.restore(checkpoint_path).assert_existing_objects_matched()
      for variable in model.variables:
        variable.assign(variable / num_checkpoints)
    else:
      reader = tf.train.load_checkpoint(checkpoint_path)
      for path in six.iterkeys(reader.get_variable_to_shape_map()):
        if not path.startswith(model_key) or ".OPTIMIZER_SLOT" in path:
          continue
        variable_path = path.replace("/.ATTRIBUTES/VARIABLE_VALUE", "")
        variable = misc.index_structure(trackables, variable_path)
        value = reader.get_tensor(path)
        variable.assign_add(value / num_checkpoints)

  new_checkpoint_manager = tf.train.CheckpointManager(checkpoint, output_dir, max_to_keep=None)
  new_checkpoint_manager.save(checkpoint_number=last_step)
  return output_dir
