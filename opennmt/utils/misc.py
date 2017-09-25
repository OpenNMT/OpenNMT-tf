"""Various utility functions to use throughout the project."""

from __future__ import print_function

import tensorflow as tf

from tensorflow.python.summary.writer.writer_cache import FileWriterCache as SummaryWriterCache


def count_lines(filename):
  """Returns the number of lines of the file `filename`."""
  with open(filename) as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def count_parameters():
  """Returns the total number of trainable parameters."""
  total = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    count = 1
    for dim in shape:
      count *= dim.value
    total += count
  return total

def get_tensor_by_name(name):
  """Gets a tensor by name in the default graph. Returns `None` if not found."""
  try:
    return tf.get_default_graph().get_tensor_by_name(name)
  except KeyError:
    return None

def extract_prefixed_keys(dictionary, prefix):
  """Returns a dictionary with all keys from `dictionary` that are prefixed
  with `prefix`.
  """
  sub_dict = {}
  for key, value in dictionary.items():
    if key.startswith(prefix):
      original_key = key[len(prefix):]
      sub_dict[original_key] = value
  return sub_dict


class LogParametersCountHook(tf.train.SessionRunHook):
  """Simple hook that logs the number of trainable parameters."""

  def begin(self):
    print("Number of trainable parameters:", count_parameters())

class WordCounterHook(tf.train.SessionRunHook):
  """Hook that counts tokens per second.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError(
          "exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir

  def begin(self):
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)

    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError(
          "Global step should be created to use WordCounterHook.")

    self._features_tensor = get_tensor_by_name("words_per_sec/features:0")
    self._labels_tensor = get_tensor_by_name("words_per_sec/labels:0")

    self._last_features_count = 0
    self._last_labels_count = 0

    if self._features_tensor is None and self._labels_tensor is None:
      raise RuntimeError(
          "Word counters should be created to use WordCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    fetches = {}
    fetches["global_step"] = self._global_step_tensor
    if self._features_tensor is not None:
      fetches["features"] = self._features_tensor
    if self._labels_tensor is not None:
      fetches["labels"] = self._labels_tensor
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):
    _ = run_context

    results = run_values.results
    global_step = results["global_step"]

    def _summarize_value(value, tag):
      if self._summary_writer is not None:
        summary = tf.Summary(value=[tf.Summary.Value(
            tag=tag, simple_value=value)])
        self._summary_writer.add_summary(summary, global_step)
      tf.logging.info("%s: %g", tag, value)

    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
          global_step)
      if elapsed_time is not None:
        if "features" in results:
          delta_features = results["features"] - self._last_features_count
          _summarize_value(delta_features / elapsed_time, "words_per_sec/features")
          self._last_features_count = results["features"]
        if "labels" in results:
          delta_labels = results["labels"] - self._last_labels_count
          _summarize_value(delta_labels / elapsed_time, "words_per_sec/labels")
          self._last_labels_count = results["labels"]
        if "features" in results and "labels" in results:
          _summarize_value(
              (delta_features + delta_labels) / elapsed_time,
              "words_per_sec/all")
