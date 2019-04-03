# pylint: disable=missing-docstring

"""Custom hooks."""

import io
import time
import six

import tensorflow as tf

from opennmt.utils import misc


class LogWordsPerSecondHook(tf.estimator.SessionRunHook):
  """Hook that logs the number of words processed per second.

  Implementation is mostly copied from StepCounterHook.
  """

  def __init__(self,
               num_words,
               global_step,
               every_n_steps=100,
               every_n_secs=None,
               output_dir=None,
               summary_writer=None):
    if (every_n_steps is None) == (every_n_secs is None):
      raise ValueError("exactly one of every_n_steps and every_n_secs should be provided.")
    self._timer = tf.estimator.SecondOrStepTimer(
        every_steps=every_n_steps,
        every_secs=every_n_secs)
    self._global_step = global_step
    self._summary_writer = summary_writer
    if self._summary_writer is None and output_dir:
      self._summary_writer = tf.compat.v1.summary.FileWriterCache.get(output_dir)
    counters = [self._create_variable(name) for name in six.iterkeys(num_words)]
    self._init_op = tf.compat.v1.variables_initializer(counters)
    self._update_op = {
        name:var.assign_add(tf.cast(count, var.dtype))
        for (name, count), var in zip(six.iteritems(num_words), counters)}
    self._last_count = [None for _ in counters]

  def _create_variable(self, name, dtype=tf.int64):
    return tf.Variable(
        initial_value=0,
        trainable=False,
        name="%s_words_counter" % name,
        dtype=dtype,
        aggregation=tf.VariableAggregation.SUM)

  def after_create_session(self, session, coord):
    _ = coord
    session.run(self._init_op)

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.estimator.SessionRunArgs([self._update_op, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    counters, step = run_values.results
    if self._timer.should_trigger_for_step(step):
      elapsed_time, _ = self._timer.update_last_triggered_step(step)
      if elapsed_time is not None:
        for i, (name, current_value) in enumerate(six.iteritems(counters)):
          if self._last_count[i] is not None:
            value = (current_value - self._last_count[i]) / elapsed_time
            tf.compat.v1.logging.info("%s_words/sec: %d", name, value)
            if self._summary_writer is not None:
              tag_name = "words_per_sec/%s" % name
              summary = tf.compat.v1.Summary(
                  value=[tf.compat.v1.Summary.Value(tag=tag_name, simple_value=value)])
              self._summary_writer.add_summary(summary, step)
          self._last_count[i] = current_value


class LogPredictionTimeHook(tf.estimator.SessionRunHook):
  """Hooks that gathers and logs prediction times."""

  def begin(self):
    self._total_time = 0
    self._total_tokens = 0
    self._total_examples = 0

  def before_run(self, run_context):
    self._run_start_time = time.time()
    predictions = run_context.original_args.fetches
    return tf.estimator.SessionRunArgs(predictions)

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
    tf.compat.v1.logging.info("Total prediction time (s): %f", self._total_time)
    tf.compat.v1.logging.info(
        "Average prediction time (s): %f", self._total_time / self._total_examples)
    if self._total_tokens > 0:
      tf.compat.v1.logging.info("Tokens per second: %f", self._total_tokens / self._total_time)


class SaveEvaluationPredictionHook(tf.estimator.SessionRunHook):
  """Hook that saves the evaluation predictions."""

  def __init__(self, model, predictions, global_step, output_file, post_evaluation_fn=None):
    """Initializes this hook.

    Args:
      model: The model for which to save the evaluation predictions.
      predictions: The predictions to save.
      global_step: The step variable.
      output_file: The output filename which will be suffixed by the current
        training step.
      post_evaluation_fn: (optional) A callable that takes as argument the
        current step and the file with the saved predictions.

    Raises:
      RuntimeError: if :obj:`predictions` is empty.
    """
    if predictions:
      raise RuntimeError("The model did not define any predictions.")
    self._model = model
    self._predictions = predictions
    self._global_step = global_step
    self._output_file = output_file
    self._post_evaluation_fn = post_evaluation_fn

  def before_run(self, run_context):  # pylint: disable=unused-argument
    return tf.estimator.SessionRunArgs([self._predictions, self._global_step])

  def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
    predictions, self._current_step = run_values.results
    self._output_path = "{}.{}".format(self._output_file, self._current_step)
    with io.open(self._output_path, encoding="utf-8", mode="a") as output_file:
      for prediction in misc.extract_batches(predictions):
        self._model.print_prediction(prediction, stream=output_file)

  def end(self, session):
    _ = session
    tf.compat.v1.logging.info("Evaluation predictions saved to %s", self._output_path)
    if self._post_evaluation_fn is not None:
      self._post_evaluation_fn(self._current_step, self._output_path)


class VariablesInitializerHook(tf.estimator.SessionRunHook):
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
    self._init_op = tf.compat.v1.variables_initializer(self._variables)

  def after_create_session(self, session, coord):
    _ = coord
    session.run(self._init_op)
