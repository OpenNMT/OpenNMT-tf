"""Checkpoint utilities."""

import os
import six

import tensorflow as tf
import numpy as np

from opennmt.utils.vocab import Vocab
from opennmt.utils.misc import count_lines


def _get_vocabulary_mapping(current_vocab_path, new_vocab_path, mode):
  """Maps vocabulary new indices to old ones. -1 means that the entry is new."""
  current_vocab = Vocab(from_file=current_vocab_path)
  new_vocab = Vocab(from_file=new_vocab_path)
  mapping = []
  if mode == "merge":
    mapping = [i for i in range(current_vocab.size)]
    for new_word in new_vocab.words:
      if current_vocab.lookup(new_word) is None:
        mapping.append(-1)
  elif mode == "replace":
    for new_word in new_vocab.words:
      idx = current_vocab.lookup(new_word)
      if idx is not None:
        mapping.append(idx)
      else:
        mapping.append(-1)
  mapping.append(current_vocab.size)  # <unk> token is always the last entry.
  return mapping

def _update_vocabulary_variable(variable, vocab_size, mapping):
  """Creates a new variable, possibly copying previous entries based on mapping."""
  dim = variable.shape.index(vocab_size)
  # Make the dimension to index the first.
  perm = list(range(len(variable.shape)))
  perm[0], perm[dim] = perm[dim], perm[0]
  variable_t = np.transpose(variable, axes=perm)
  new_shape = list(variable_t.shape)
  new_shape[0] = len(mapping)
  new_variable_t = np.zeros(new_shape, dtype=variable.dtype)
  for i, j in enumerate(mapping):
    if j >= 0:
      new_variable_t[i] = variable_t[j]
  new_variable = np.transpose(new_variable_t, axes=perm)
  return new_variable

def _update_vocabulary_variables(variables, current_vocab_path, new_vocab_path, scope_name, mode):
  current_size = count_lines(current_vocab_path) + 1
  mapping = _get_vocabulary_mapping(current_vocab_path, new_vocab_path, mode)
  for name, tensor in six.iteritems(variables):
    if scope_name in name and any(d == current_size for d in tensor.shape):
      tf.logging.debug("Updating variable %s" % name)
      variables[name] = _update_vocabulary_variable(tensor, current_size, mapping)

def _variable_is_trainable(name, value):
  _ = name
  return value.dtype not in (np.int32, np.int64)  # Assume that int variables are not trainable.

def _create_checkpoint_from_variables(variables, output_dir, latest_step=None, session_config=None):
  # The straightforward approach would be to create new variables using a
  # constant_initializer. However, this would save the constant value in the
  # checkpoint meta file which would increase its size dramatically. Instead, we
  # create variables with their default initializer but run an assignment op
  # that writes the new value. Inspired by:
  # github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/bin/t2t_avg_all.py
  if "global_step" in variables:
    latest_step = variables["global_step"]
    del variables["global_step"]
  tf_vars = [
      tf.get_variable(
          name,
          shape=value.shape,
          dtype=tf.as_dtype(value.dtype),
          trainable=_variable_is_trainable(name, value))
      for name, value in six.iteritems(variables)]
  placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
  assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, placeholders)]

  out_base_file = os.path.join(output_dir, "model.ckpt")
  global_step = tf.get_variable(
      "global_step",
      initializer=tf.constant(latest_step, dtype=tf.int64),
      trainable=False)
  saver = tf.train.Saver(tf.global_variables())

  with tf.Session(config=session_config) as sess:
    sess.run(tf.global_variables_initializer())
    for p, assign_op, (name, value) in zip(placeholders, assign_ops, six.iteritems(variables)):
      sess.run(assign_op, {p: value})
    tf.logging.info("Saving new checkpoint to %s" % output_dir)
    saver.save(sess, out_base_file, global_step=global_step)

  return output_dir

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

def convert_checkpoint(checkpoint_path,
                       output_dir,
                       source_dtype,
                       target_type,
                       session_config=None):
  """Converts checkpoint variables from one dtype to another.

  Args:
    checkpoint_path: The path to the checkpoint to convert.
    output_dir: The directory that will contain the converted checkpoint.
    source_dtype: The data type to convert from.
    target_dtype: The data type to convert to.
    session_config: Optional configuration to use when creating the session.

  Returns:
    The path to the directory containing the converted checkpoint.

  Raises:
    ValueError: if :obj:`output_dir` points to the same directory as
      :obj:`checkpoint_path`.
  """
  if os.path.dirname(checkpoint_path) == output_dir:
    raise ValueError("Checkpoint and output directory must be different")
  variables = get_checkpoint_variables(checkpoint_path)
  for name, value in six.iteritems(variables):
    if not name.startswith("optim") and tf.as_dtype(value.dtype) == source_dtype:
      variables[name] = value.astype(target_type.as_numpy_dtype())
  return _create_checkpoint_from_variables(
      variables,
      output_dir,
      session_config=session_config)

def update_vocab(model_dir,
                 output_dir,
                 current_src_vocab,
                 current_tgt_vocab,
                 new_src_vocab=None,
                 new_tgt_vocab=None,
                 mode="merge",
                 session_config=None):
  """Updates the last checkpoint to support new vocabularies.

  This allows to add new words to a model while keeping the previously learned
  weights.

  Args:
    model_dir: The directory containing checkpoints (the most recent will be loaded).
    output_dir: The directory that will contain the converted checkpoint.
    current_src_vocab: Path to the source vocabulary currently use in the model.
    current_tgt_vocab: Path to the target vocabulary currently use in the model.
    new_src_vocab: Path to the new source vocabulary to support.
    new_tgt_vocab: Path to the new target vocabulary to support.
    mode: Update mode: "merge" keeps all existing words and adds new words,
      "replace" makes the new vocabulary file the active one. In all modes,
      if an existing word appears in the new vocabulary, its learned weights
      are kept in the converted checkpoint.
    session_config: Optional configuration to use when creating the session.

  Returns:
    The path to the directory containing the converted checkpoint.

  Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir` or if
      :obj:`mode` is invalid.
  """
  if model_dir == output_dir:
    raise ValueError("Model and output directory must be different")
  if mode not in ("merge", "replace"):
    raise ValueError("invalid vocab update mode: %s" % mode)
  if new_src_vocab is None and new_tgt_vocab is None:
    return model_dir
  checkpoint_path = tf.train.latest_checkpoint(model_dir)
  tf.logging.info("Updating vocabulary related variables in checkpoint %s" % checkpoint_path)
  variable_value = get_checkpoint_variables(checkpoint_path)
  if new_src_vocab is not None:
    _update_vocabulary_variables(variable_value, current_src_vocab, new_src_vocab, "encoder", mode)
  if new_tgt_vocab is not None:
    _update_vocabulary_variables(variable_value, current_tgt_vocab, new_tgt_vocab, "decoder", mode)
  return _create_checkpoint_from_variables(
      variable_value,
      output_dir,
      session_config=session_config)


def average_checkpoints(model_dir, output_dir, max_count=8, session_config=None):
  """Averages checkpoints.

  Args:
    model_dir: The directory containing checkpoints.
    output_dir: The directory that will contain the averaged checkpoint.
    max_count: The maximum number of checkpoints to average.
    session_config: Configuration to use when creating the session.

  Returns:
    The path to the directory containing the averaged checkpoint.

  Raises:
    ValueError: if :obj:`output_dir` is the same as :obj:`model_dir`.
  """
  if model_dir == output_dir:
    raise ValueError("Model and output directory must be different")

  checkpoints_path = tf.train.get_checkpoint_state(model_dir).all_model_checkpoint_paths
  if len(checkpoints_path) > max_count:
    checkpoints_path = checkpoints_path[-max_count:]
  num_checkpoints = len(checkpoints_path)

  tf.logging.info("Averaging %d checkpoints..." % num_checkpoints)
  tf.logging.info("Listing variables...")

  new_variables = {}
  for i, checkpoint_path in enumerate(checkpoints_path):
    tf.logging.info("Loading checkpoint %s" % checkpoint_path)
    variables = get_checkpoint_variables(checkpoint_path)
    for name, value in six.iteritems(variables):
      if _variable_is_trainable(name, value):
        scaled_value = value / num_checkpoints
        if name in new_variables:
          new_variables[name] += scaled_value
        else:
          new_variables[name] = scaled_value
      elif i + 1 == num_checkpoints:  # Take non trainable variables from the last checkpoint.
        new_variables[name] = value

  return _create_checkpoint_from_variables(
      new_variables,
      output_dir,
      session_config=session_config)
