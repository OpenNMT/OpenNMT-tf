"""Training related classes and functions."""

import abc
import contextlib
import time

import tensorflow as tf

from opennmt.optimizers import utils as optimizer_util
from opennmt.utils import misc


class Trainer(abc.ABC):
  """Base class for model trainer."""

  def __init__(self, checkpoint, is_master=True):
    """Initializes the trainer.

    Args:
      checkpoint: A :class:`opennmt.utils.Checkpoint` instance.
      is_master: Whether this trainer instance is the master trainer.
    """
    self._checkpoint = checkpoint
    self._is_master = is_master
    self._model = checkpoint.model
    self._summary_writer = tf.summary.create_file_writer(checkpoint.model_dir)

    optimizer = checkpoint.optimizer
    if optimizer is None:
      raise ValueError("No optimizer is defined")
    graph_optimizer_options = tf.config.optimizer.get_experimental_options()
    mixed_precision_enabled = graph_optimizer_options.get("auto_mixed_precision")
    if (mixed_precision_enabled
        and not isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)):
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
    self._optimizer = optimizer

  def __call__(self,
               dataset,
               max_step=None,
               accum_steps=1,
               report_steps=100,
               save_steps=5000,
               evaluator=None,
               eval_steps=5000,
               moving_average_decay=None):
    """Runs the training.

    Args:
      dataset: A ``tf.data.Dataset`` or a function taking a ``tf.distribute.InputContext``
        instance and returning a ``tf.data.Dataset``.
      max_step: The final training step.
      accum_steps: The number of gradient accumulation steps.
      report_steps: Report status every this many steps.
      save_steps: Save a checkpoint every this many steps.
      evaluator: A :class:`opennmt.evaluation.Evaluator` instance to call for
        evaluation.
      eval_steps: Evaluate every this many steps.
      moving_average_decay: If set, maintain an exponential moving average of the model
        variables using this decay value (usually close to 1, e.g. 0.9999). See
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.
    """
    if max_step is not None and self._optimizer.iterations.numpy() >= max_step:
      tf.get_logger().warning("Model already reached max_step = %d. Exiting.", max_step)
      return
    if evaluator is not None and evaluator.should_stop():
      tf.get_logger().warning("Early stopping conditions are already met. Exiting.")
      return

    with self._summary_writer.as_default():
      iterations = self._optimizer.iterations
      tf.summary.experimental.set_step(iterations)

      moving_average = None
      last_report_step = iterations.numpy()
      last_report_time = time.time()
      for loss in self._steps(dataset, accum_steps=accum_steps, report_steps=report_steps):
        if tf.math.is_nan(loss):
          raise RuntimeError("Model diverged with loss = NaN.")

        if moving_average_decay is not None and self._is_master:
          if moving_average is None:
            moving_average = MovingAverage(
                self._model.trainable_variables,
                iterations,
                decay=moving_average_decay)
          else:
            moving_average.update()

        step = iterations.numpy()
        if step % report_steps == 0:
          _report_training_status(
              step,
              loss,
              self._optimizer.learning_rate,
              self._get_words_counters(),
              last_report_step,
              last_report_time)
          last_report_step = step
          last_report_time = time.time()
        if step == 1 or (save_steps is not None and step % save_steps == 0):
          self._save_checkpoint(step, moving_average=moving_average)
        if eval_steps is not None and step % eval_steps == 0:
          early_stop = self._evaluate(evaluator, step, moving_average=moving_average)
          if early_stop:
            tf.get_logger().warning("Early stopping conditions are met. Exiting.")
            break
        if step == max_step:
          break

      self._save_checkpoint(step, moving_average=moving_average)
      self._evaluate(evaluator, step, moving_average=moving_average)

  @abc.abstractmethod
  def _steps(self, dataset, accum_steps=1, report_steps=None):
    """Returns a generator over training steps (i.e. parameters update).

    Args:
      dataset: The training dataset.
      accum_steps: Accumulate the gradients of this many steps/batches.
      report_steps: Report summary statistics every this many steps. This should
        typically be used in a ``tf.summary.record_if`` context.

    Returns:
      A generator that yields a loss value to report for this step.
    """
    raise NotImplementedError()

  def _get_words_counters(self):
    """Returns the accumulated words counters and resets them.

    This is used to report the words per second in the training logs.

    Returns:
      A dictionary mapping a counter name to a Python value.
    """
    return {}

  def _run_model(self, source, target):
    """Computes the loss of the given source and target pair.

    Args:
      source: A nested structure of tensors.
      target: A nested structure of tensors.

    Returns:
      A tuple containing,

      - The loss to compute the gradients.
      - The loss to report.
    """
    first_call = not self._model.built
    outputs, _ = self._model(
        source,
        labels=target,
        training=True,
        step=self._optimizer.iterations)
    loss = self._model.compute_loss(outputs, target, training=True)
    if isinstance(loss, tuple):
      training_loss = loss[0] / loss[1]
      reported_loss = loss[0] / loss[2] if len(loss) > 2 else training_loss
    else:
      training_loss, reported_loss = loss, loss
    training_loss = self._model.regularize_loss(
        training_loss, variables=self._model.trainable_variables)
    if first_call and self._is_master:
      self._model.visualize(self._checkpoint.model_dir)
      tf.get_logger().info("Number of model parameters: %d", self._model.count_params())
      tf.get_logger().info(
          "Number of model weights: %d (trainable = %d, non trainable = %d)",
          len(self._model.weights),
          len(self._model.trainable_weights),
          len(self._model.non_trainable_weights))
    return training_loss, reported_loss

  def _save_checkpoint(self, step, moving_average=None):
    """Saves a checkpoint for step."""
    if not self._is_master or step == self._checkpoint.last_saved_step:
      return
    with moving_average.shadow_variables() if moving_average is not None else contextlib.suppress():
      self._checkpoint.save(step)

  def _evaluate(self, evaluator, step, moving_average=None):
    """Runs evaluation for step. Returns ``True`` is early conditions are met."""
    if not self._is_master or evaluator is None or step == evaluator.last_evaluated_step:
      return False
    with moving_average.shadow_variables() if moving_average is not None else contextlib.suppress():
      evaluator(step)
      return evaluator.should_stop()


class BasicTrainer(Trainer):
  """Basic single GPU trainer."""

  def _steps(self, dataset, accum_steps=1, report_steps=None):
    if accum_steps != 1:
      raise ValueError("BasicTrainer does not support gradient accumulation")
    if callable(dataset):
      dataset = dataset(tf.distribute.InputContext())

    @tf.function(input_signature=dataset.element_spec)
    def _step(source, target):
      training_loss, reported_loss = self._run_model(source, target)
      variables = self._model.trainable_variables
      gradients = self._optimizer.get_gradients(training_loss, variables)
      self._optimizer.apply_gradients(list(zip(gradients, variables)))
      return reported_loss

    for source, target in dataset:
      yield _step(source, target)


class DistributionStrategyTrainer(Trainer):
  """Trainer based on distribution strategies."""

  def __init__(self, checkpoint, devices=None):
    """Initializes the trainer.

    Args:
      checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance.
      devices: List of device strings to use for training.
    """
    super(DistributionStrategyTrainer, self).__init__(checkpoint)
    if not devices:
      devices = misc.get_devices(count=1)  # Train with 1 device by default.
    self._strategy = tf.distribute.MirroredStrategy(devices=devices)
    self._words_counters = {}
    with self._strategy.scope():
      # Create some variables under the strategy scope.
      _ = self._optimizer.iterations
      self._gradient_accumulator = optimizer_util.GradientAccumulator()

  def _get_words_counters(self):
    return {name:value.numpy() for name, value in self._synchronize_words_counters().items()}

  def _steps(self, dataset, accum_steps=1, report_steps=None):
    self._gradient_accumulator.reset()
    self._words_counters.clear()
    for i, loss in enumerate(self._accumulate_next_gradients(dataset, report_steps=report_steps)):
      if tf.math.is_nan(loss):
        raise RuntimeError("Model diverged with loss = NaN.")
      if i == 0 or (i + 1) % accum_steps == 0:
        self._apply_gradients()
        yield loss

  def _accumulate_next_gradients(self, dataset, report_steps=None):
    """Accumulates the gradients from the next element in :obj:`dataset`."""

    # We prefer not to use experimental_distribute_dataset here because it
    # sometimes fails to split the batches (noticed with tokens batch type).
    # We also assume for now that we are training with a single worker
    # otherwise we would need to correctly shard the input dataset.
    dataset_fn = dataset if callable(dataset) else lambda _: dataset
    distributed_dataset = self._strategy.experimental_distribute_datasets_from_function(
        dataset_fn)

    # Get the next element within the tf.function for more pipelining.
    # See: https://github.com/tensorflow/tensorflow/issues/29075#issuecomment-513390242
    iterator = iter(distributed_dataset)

    @tf.function
    def _accumulate_next():
      if report_steps is None:
        should_record_summaries = False
      else:
        should_record_summaries = tf.logical_and(
            tf.equal(self._optimizer.iterations % report_steps, 0),
            tf.equal(self._gradient_accumulator.step, 0))
      with tf.summary.record_if(should_record_summaries):
        per_replica_source, per_replica_target = next(iterator)
        return self._accumulate_gradients(
            per_replica_source, per_replica_target, should_record_summaries)

    while True:
      try:
        yield _accumulate_next()
      except tf.errors.OutOfRangeError:
        break

  def _accumulate_gradients(self, per_replica_source, per_replica_target, should_record_summaries):
    """Accumulates the gradients (cross-replica)."""
    per_replica_loss = self._strategy.experimental_run_v2(
        self._accumulate_gradients_on_replica,
        args=(per_replica_source, per_replica_target, should_record_summaries))
    # TODO: this reduction could be delayed until _step is called.
    return self._strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

  def _accumulate_gradients_on_replica(self, source, target, should_record_summaries):
    """Accumulates the gradients (in replica)."""
    training_loss, reported_loss = self._run_model(source, target)
    variables = self._model.trainable_variables
    gradients = self._optimizer.get_gradients(training_loss, variables)
    self._gradient_accumulator(gradients)
    _summarize_gradients(gradients, should_record_summaries)
    self._update_words_counter("source", source)
    self._update_words_counter("target", target)
    return reported_loss

  def _update_words_counter(self, name, features):
    """Accumulates number of source and target tokens to report throughput."""
    length = features.get("length")
    if length is None:
      return
    num_words = tf.reduce_sum(length)
    counter = self._words_counters.get(name)
    if counter is None:
      counter = tf.Variable(
          tf.constant(0, dtype=tf.int64),
          trainable=False,
          synchronization=tf.VariableSynchronization.ON_READ,
          aggregation=tf.VariableAggregation.SUM)
      self._words_counters[name] = counter
    counter.assign_add(tf.cast(num_words, tf.int64))

  @tf.function
  def _synchronize_words_counters(self):
    """Synchronizes and resets words counters values across replicas."""
    sync_words_counters = {
        name:counter.read_value() for name, counter in self._words_counters.items()}
    self._strategy.experimental_run_v2(self._reset_words_counters_on_replica)
    return sync_words_counters

  def _reset_words_counters_on_replica(self):
    """Resets the variables that count words (in replica)."""
    for counter in self._words_counters.values():
      counter.assign(tf.constant(0, dtype=tf.int64))

  @tf.function
  def _apply_gradients(self):
    """Applies the gradients (cross-replica)."""
    self._strategy.experimental_run_v2(self._apply_gradients_on_replica)

  def _apply_gradients_on_replica(self):
    """Applies the gradients (in replica)."""
    variables = self._model.trainable_variables
    # optimizer.apply_gradients will sum the gradients accross replicas.
    gradient_scale = self._gradient_accumulator.step * self._strategy.num_replicas_in_sync
    grads_and_vars = [
        (gradient / tf.cast(gradient_scale, gradient.dtype), variable)
        for gradient, variable in zip(self._gradient_accumulator.gradients, variables)]
    self._optimizer.apply_gradients(grads_and_vars)
    self._gradient_accumulator.reset()


def _report_training_status(step,
                            loss,
                            learning_rate,
                            words_counters,
                            last_report_step,
                            last_report_time):
  elapsed_time = time.time() - last_report_time

  steps_per_sec = (step - last_report_step) / elapsed_time
  tf.summary.scalar("steps_per_sec", steps_per_sec, description="Training steps per second")
  steps_per_sec_fmt = "steps/s = %0.2f" % steps_per_sec

  words_per_sec_fmt = []
  for name, counter in words_counters.items():
    avg = int(counter / elapsed_time)
    tf.summary.scalar(
        "words_per_sec/%s" % name,
        avg,
        description="%s words per second" % name.capitalize())
    words_per_sec_fmt.append("%s words/s = %d" % (name, avg))

  if isinstance(learning_rate, tf.optimizers.schedules.LearningRateSchedule):
    learning_rate = learning_rate(step)
  elif isinstance(learning_rate, tf.Variable):
    learning_rate = learning_rate.value()

  tf.get_logger().info(
      "Step = %d ; %s ; Learning rate = %f ; Loss = %f",
      step,
      ", ".join([steps_per_sec_fmt] + list(sorted(words_per_sec_fmt))),
      learning_rate,
      loss)
  tf.summary.scalar("loss", loss, description="Training loss")
  tf.summary.scalar("optim/learning_rate", learning_rate, description="Learning rate")

def _summarize_gradients(gradients, should_record):
  # Only compute the gradients global norm when the value is actually recorded.
  tf.summary.scalar(
      "gradients/global_norm",
      tf.cond(
          should_record,
          true_fn=lambda: tf.linalg.global_norm(gradients),
          false_fn=lambda: tf.constant(0, dtype=gradients[0].dtype)))


class MovingAverage(object):
  """Object holding an exponential moving average of variables."""

  def __init__(self, variables, step, decay=0.9999):
    """Initializes the moving average object.

    Args:
      variables: The list of variable for which to maintain a moving average.
      step: The training step counter as a ``tf.Variable``.
      decay: The decay rate of the exponential moving average. Usually close to
        1, e.g. 0.9999, see the complete formula on
        https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage.

    Raises:
      TypeError: is :obj:`step` is not a ``tf.Variable``.
    """
    if not isinstance(step, tf.Variable):
      raise TypeError("step should be a tf.Variable")
    if decay < 0.9 or decay > 1:
      tf.get_logger().warning("Moving average decay should be close to 1 (e.g. 0.9999) but you "
                              "passed %f, is it correct? See https://www.tensorflow.org/api_docs"
                              "/python/tf/train/ExponentialMovingAverage for details about the "
                              "formula and recommended decay values.")
    self._ema = tf.train.ExponentialMovingAverage(decay, num_updates=step)
    self._variables = variables
    self.update()

  @tf.function
  def update(self):
    """Updates the moving average of the variables."""
    self._ema.apply(var_list=list(map(misc.get_primary_variable, self._variables)))

  @contextlib.contextmanager
  def shadow_variables(self):
    """Returns a context manager that assigns the variables to their moving
    average value on enter and restores the previous value on exit.

    Returns:
      A context manager.
    """
    # TODO: Do we want to shadow the values on all replicas?
    previous_values = []
    for variable in self._variables:
      previous_values.append(variable.value())
      variable.assign(self._ema.average(misc.get_primary_variable(variable)))
    yield
    for previous_value, variable in zip(previous_values, self._variables):
      variable.assign(previous_value)
