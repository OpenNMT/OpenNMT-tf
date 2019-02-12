# pylint: disable=missing-docstring

"""Custom hooks."""

from __future__ import print_function

import io
import time
import six

import tensorflow as tf

from opennmt.utils import compat, misc

_SESSION_RUN_HOOK = compat.tf_compat(v2="estimator.SessionRunHook", v1="train.SessionRunHook")


class LogParametersCountHook(_SESSION_RUN_HOOK):
  """Simple hook that logs the number of trainable parameters."""

  def begin(self):
    tf.logging.info("Number of trainable parameters: %d", misc.count_parameters())


_DEFAULT_COUNTERS_COLLECTION = "counters"


def add_counter(name, tensor):
  """Registers a new counter.

  Args:
    name: The name of this counter.
    tensor: The integer ``tf.Tensor`` to count.

  Returns:
    An op that increments the counter.

  See Also:
    :meth:`opennmt.utils.misc.WordCounterHook` that fetches these counters
    to log their value in TensorBoard.
  """
  count = tf.cast(tensor, tf.int64)
  total_count_init = tf.Variable(
      initial_value=0,
      name=name + "_init",
      trainable=False,
      dtype=count.dtype)
  total_count = tf.assign_add(
      total_count_init,
      count,
      name=name)
  tf.add_to_collection(_DEFAULT_COUNTERS_COLLECTION, total_count)
  return total_count


class CountersHook(_SESSION_RUN_HOOK):
  """Hook that summarizes counters.

  Implementation is mostly copied from StepCounterHook.

  Deprecated: use LogWordsPerSecondHook instead.
  """

  def __init__(self,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None,
               counters=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.train.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)

    self._summary_writer = summary_writer
    self._output_dir = output_dir
    self._counters = counters

  def begin(self):
    if self._counters is None:
      self._counters = tf.get_collection(_DEFAULT_COUNTERS_COLLECTION)
    if not self._counters:
      return

    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)

    self._last_count = [None for _ in self._counters]
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use WordCounterHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if not self._counters:
      return None
    return tf.train.SessionRunArgs([self._counters, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if not self._counters:
      return

    counters, step = run_values.results
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i in range(len(self._counters)):
          if self._last_count[i] is not None:
            name = self._counters[i].name.split(":")[0]
            value = (counters[i] - self._last_count[i]) / elapsed_time
            if self._summary_writer is not None:
              summary = tf.Summary(value=[tf.Summary.Value(tag=name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
            tf.logging.info("%s: %g", name, value)
          self._last_count[i] = counters[i]


class LogWordsPerSecondHook(_SESSION_RUN_HOOK):
  """Hook that logs the number of words processed per second.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               num_words,
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
    self._num_words = num_words

  def _create_variable(self, name, dtype=tf.int64):
    return tf.Variable(
        initial_value=0,
        trainable=False,
        collections=[],
        name="%s_words_counter" % name,
        dtype=dtype)

  def begin(self):
    if not self._num_words:
      return
    if self._summary_writer is None and self._output_dir:
      self._summary_writer = tf.summary.FileWriterCache.get(self._output_dir)
    counters = [self._create_variable(name) for name in six.iterkeys(self._num_words)]
    self._init_op = tf.variables_initializer(counters)
    self._update_op = {
        name:tf.assign_add(var, tf.cast(count, tf.int64))
        for (name, count), var in zip(six.iteritems(self._num_words), counters)}
    self._last_count = [None for _ in counters]
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use LogWordsPerSecondHook.")

  def after_create_session(self, session, coord):
    _ = coord
    if self._num_words:
      session.run(self._init_op)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    if not self._num_words:
      return None
    return tf.train.SessionRunArgs([self._update_op, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    if not self._num_words:
      return

    counters, step = run_values.results
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i, (name, current_value) in enumerate(six.iteritems(counters)):
          if self._last_count[i] is not None:
            value = (current_value - self._last_count[i]) / elapsed_time
            tf.logging.info("%s_words/sec: %d", name, value)
            if self._summary_writer is not None:
              tag_name = "words_per_sec/%s" % name
              summary = tf.Summary(value=[tf.Summary.Value(tag=tag_name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
          self._last_count[i] = current_value


class LogPredictionTimeHook(_SESSION_RUN_HOOK):
  """Hooks that gathers and logs prediction times."""

  def begin(self):
    self._total_time = 0
    self._total_tokens = 0
    self._total_examples = 0

  def before_run(self, run_context):
    self._run_start_time = time.time()
    predictions = run_context.original_args.fetches
    return tf.train.SessionRunArgs(predictions)

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    self._total_time += time.time() - self._run_start_time
    predictions = run_values.results
    batch_size = next(six.itervalues(predictions)).shape[0]
    self._total_examples += batch_size
    length = predictions.get("length")
    if length is not None:
      if len(length.shape) == 2:
        length = length[:, 0]
      self._total_tokens += sum(length)

  def end(self, session):
    _ = session
    tf.logging.info("Total prediction time (s): %f", self._total_time)
    tf.logging.info("Average prediction time (s): %f", self._total_time / self._total_examples)
    if self._total_tokens > 0:
      tf.logging.info("Tokens per second: %f", self._total_tokens / self._total_time)


class SaveEvaluationPredictionHook(_SESSION_RUN_HOOK):
  """Hook that saves the evaluation predictions."""

  def __init__(self, model, output_file, post_evaluation_fn=None, predictions=None):
    """Initializes this hook.

    Args:
      model: The model for which to save the evaluation predictions.
      output_file: The output filename which will be suffixed by the current
        training step.
      post_evaluation_fn: (optional) A callable that takes as argument the
        current step and the file with the saved predictions.
      predictions: The predictions to save.
    """
    self._model = model
    self._output_file = output_file
    self._post_evaluation_fn = post_evaluation_fn
    self._predictions = predictions

  def begin(self):
    if self._predictions is None:
      self._predictions = misc.get_dict_from_collection("predictions")
    if not self._predictions:
      raise RuntimeError("The model did not define any predictions.")
    self._global_step = tf.train.get_global_step()
    if self._global_step is None:
      raise RuntimeError("Global step should be created to use SaveEvaluationPredictionHook.")

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.train.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    predictions, self._current_step = run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    with io.open(self._output_path, encoding="utf-8", mode="a") as output_file:
      for prediction in misc.extract_batches(predictions):
        self._model.print_prediction(prediction, stream=output_file)

  def end(self, session):
    _ = session
    tf.logging.info("Evaluation predictions saved to %s", self._output_path)
    if self._post_evaluation_fn is not None:
      self._post_evaluation_fn(self._current_step, self._output_path)


class LoadWeightsFromCheckpointHook(_SESSION_RUN_HOOK):
  """"Hook that loads model variables from checkpoint before starting the training."""

  def __init__(self, checkpoint_path):
    self.checkpoint_path = checkpoint_path

  def begin(self):
    var_list = tf.train.list_variables(self.checkpoint_path)

    names = []
    for name, _ in var_list:
      if (not name.startswith("optim")
          and not name.startswith("global_step")
          and not name.startswith("words_per_sec")):
        names.append(name)

    self.values = {}
    reader = tf.train.load_checkpoint(self.checkpoint_path)
    for name in names:
      self.values[name] = reader.get_tensor(name)

    tf_vars = []
    current_scope = tf.get_variable_scope()
    reuse = tf.AUTO_REUSE if hasattr(tf, "AUTO_REUSE") else True
    with tf.variable_scope(current_scope, reuse=reuse):
      for name, value in six.iteritems(self.values):
        tf_vars.append(tf.get_variable(name, shape=value.shape, dtype=tf.as_dtype(value.dtype)))

    self.placeholders = [tf.placeholder(v.dtype, shape=v.shape) for v in tf_vars]
    self.assign_ops = [tf.assign(v, p) for (v, p) in zip(tf_vars, self.placeholders)]

  def after_create_session(self, session, coord):
    _ = coord
    for p, op, value in zip(self.placeholders, self.assign_ops, six.itervalues(self.values)):
      session.run(op, {p: value})


class VariablesInitializerHook(_SESSION_RUN_HOOK):
  """Hook that initializes some variables in the current session. This is useful
  when using internal variables (e.g. for value accumulation) that are not saved
  in the checkpoints.
  """

  def __init__(self, variables):
    """Initializes this hook.

    Args:
      variables: A list of variables to initialize.
    """
    self._variables = variables
    self._init_op = None

  def begin(self):
    self._init_op = tf.variables_initializer(self._variables)

  def after_create_session(self, session, coord):
    _ = coord
    session.run(self._init_op)
