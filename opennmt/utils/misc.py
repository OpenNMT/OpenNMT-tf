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
    tf.logging.info("Number of trainable parameters: %d", count_parameters())

class CountersHook(tf.train.SessionRunHook):
  """Hook that summarizes counters.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir

  def begin(self):
    self._counters = tf.get_collection("counters")
    if not self._counters:
      return

    if self._summary_writer is None and self._output_dir:
      self._summary_writer = SummaryWriterCache.get(self._output_dir)

    self._last_count = [0 for _ in self._counters]
    self._global_step_tensor = tf.train.get_global_step()
    if self._global_step_tensor is None:
      raise RuntimeError("Global step should be created to use WordCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if not self._counters:
      return None
    fetches = list(self._counters) + [self._global_step_tensor]
    return tf.train.SessionRunArgs(fetches)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if not self._counters:
      return

    results = run_values.results
    global_step = results.pop()

    if self._timer.should_trigger_for_step(global_step):
      elapsed_time, _ = self._timer.update_last_triggered_step(global_step)
      if elapsed_time is not None:
        for i in range(len(self._counters)):
          name = self._counters[i].name
          value = (results[i] - self._last_count[i]) / elapsed_time
          self._last_count[i] = results[i]
          if self._summary_writer is not None:
            summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
            self._summary_writer.add_summary(summary, global_step)
          tf.logging.info("%s: %g", name, value)
