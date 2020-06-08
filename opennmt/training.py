"""Training related classes and functions."""

import contextlib
import itertools
import time

import tensorflow as tf

from opennmt.optimizers import utils as optimizer_util
from opennmt.utils import misc


class Trainer:
  """Base class for model trainer, implementing single-GPU training."""

  def __init__(self, model, optimizer, checkpoint=None, is_master=True):
    """Initializes the trainer.

    Args:
      model: A :class:`opennmt.models.Model` instance to train.
      optimizer: A ``tf.keras.optimizers.Optimizer`` instance.
      checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance. If
        not set, no checkpoints will be saved.
      is_master: Whether this trainer instance is the master trainer.
    """
    self._checkpoint = checkpoint
    self._is_master = is_master
    self._model = model
    if checkpoint is not None:
      self._summary_writer = tf.summary.create_file_writer(checkpoint.model_dir)
    else:
      self._summary_writer = tf.summary.create_noop_writer()
    self._words_counters = {}
    self._gradient_accumulator = optimizer_util.GradientAccumulator()

    if optimizer is None:
      raise ValueError("No optimizer is defined")
    graph_optimizer_options = tf.config.optimizer.get_experimental_options()
    mixed_precision_enabled = graph_optimizer_options.get("auto_mixed_precision")
    if (mixed_precision_enabled
        and not isinstance(optimizer, tf.keras.mixed_precision.experimental.LossScaleOptimizer)):
      optimizer = _LossScaleOptimizer(optimizer, "dynamic")
    self._optimizer = optimizer

  @property
  def num_replicas(self):
    """Number of synchronous training replicas."""
    return 1

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

    self._gradient_accumulator.reset()
    self._words_counters.clear()

    with self._summary_writer.as_default():
      iterations = self._optimizer.iterations
      tf.summary.experimental.set_step(iterations)

      step = None
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
          words_counters = self._get_words_counters()
          if self._is_master:
            _report_training_status(
                step,
                loss,
                self._optimizer.learning_rate,
                words_counters,
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

      if step is None:
        raise RuntimeError("No training steps were executed. This usually means the "
                           "training file is empty or all examples were filtered out. "
                           "For the latter, verify that the maximum_*_length values are "
                           "consistent with your data.")
      self._save_checkpoint(step, moving_average=moving_average)
      self._evaluate(evaluator, step, moving_average=moving_average)

  def _save_checkpoint(self, step, moving_average=None):
    """Saves a checkpoint for step."""
    if not self._is_master or self._checkpoint is None or step == self._checkpoint.last_saved_step:
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

  def _finalize_dataset(self, dataset):
    """Returns the final dataset instance to be used for training.

    Args:
      dataset: A ``tf.data.Dataset`` or a function taking a ``tf.distribute.InputContext``
        instance and returning a ``tf.data.Dataset``.

    Returns:
      A ``tf.data.Dataset``.
    """
    if callable(dataset):
      dataset = dataset(tf.distribute.InputContext())
    return dataset

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
    dataset = self._finalize_dataset(dataset)
    iterator = iter(dataset)

    # Wrap forward and step with tf.function.
    # We get the next dataset element within the function for increased efficiency.

    if accum_steps is None or accum_steps == 1:

      @tf.function
      def _forward_and_step():
        source, target = next(iterator)
        return self._forward_and_step(
            source,
            target,
            record_summaries=self._should_record_summaries(report_steps, with_accum=False))

      for i in itertools.count():
        try:
          loss = _forward_and_step()
        except tf.errors.OutOfRangeError:  # Dataset iterator exhausted.
          break
        if i == 0:
          self._broadcast_variables()
        yield loss

    else:

      @tf.function
      def _forward():
        source, target = next(iterator)
        return self._forward(
            source,
            target,
            record_summaries=self._should_record_summaries(report_steps, with_accum=True))

      @tf.function
      def _step():
        return self._step()

      for i in itertools.count():
        try:
          loss = _forward()
        except tf.errors.OutOfRangeError:  # Dataset iterator exhausted.
          break
        if i == 0 or (i + 1) % accum_steps == 0:
          _step()
          if i == 0:
            self._broadcast_variables()
          yield loss

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
    self._update_words_counter("source", source)
    if not self._model.unsupervised:
      self._update_words_counter("target", target)
    if first_call and self._is_master:
      if self._checkpoint is not None:
        self._model.visualize(self._checkpoint.model_dir)
      tf.get_logger().info("Number of model parameters: %d", self._model.count_params())
      tf.get_logger().info(
          "Number of model weights: %d (trainable = %d, non trainable = %d)",
          len(self._model.weights),
          len(self._model.trainable_weights),
          len(self._model.non_trainable_weights))
    return training_loss, reported_loss

  def _should_record_summaries(self, report_steps, with_accum=False):
    """Returns a boolean tensor to be used in tf.summary.record_if."""
    if report_steps is None or not self._is_master:
      return False
    record_summaries = tf.equal(self._optimizer.iterations % report_steps, 0)
    if with_accum:
      record_summaries = tf.logical_and(
          record_summaries,
          tf.equal(self._gradient_accumulator.step, 0))
    return record_summaries

  def _compute_gradients(self, source, target, record_summaries=False):
    """Computes the gradient of a training example."""
    with tf.summary.record_if(record_summaries):
      training_loss, reported_loss = self._run_model(source, target)
      gradients = self._optimizer.get_gradients(training_loss, self._model.trainable_variables)
      _summarize_gradients(gradients, record_summaries)
    return reported_loss, gradients

  def _apply_gradients(self, gradients, scale=1):
    """Applies the gradients."""
    gradient_scale = scale * self.num_replicas
    if tf.is_tensor(gradient_scale) or gradient_scale != 1:
      gradients = [
          self._all_reduce_sum(gradient / tf.cast(gradient_scale, gradient.dtype))
          for gradient in gradients]
    self._optimizer.apply_gradients(list(zip(gradients, self._model.trainable_variables)))

  def _forward_and_step(self, source, target, record_summaries=False):
    """Forwards a training example and applies the gradients, without accumulation."""
    loss, gradients = self._compute_gradients(source, target, record_summaries=record_summaries)
    self._apply_gradients(gradients)
    return loss

  def _forward(self, source, target, record_summaries=False):
    """Forwards a training example and accumulates the gradients."""
    loss, gradients = self._compute_gradients(source, target, record_summaries=record_summaries)
    self._gradient_accumulator(gradients)
    return loss

  def _step(self):
    """Applies gradients and resets accumulation."""
    self._apply_gradients(
        self._gradient_accumulator.gradients,
        scale=self._gradient_accumulator.step)
    self._gradient_accumulator.reset()

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
  def _get_words_counters(self):
    """Returns the accumulated words counters and resets them.

    This is used to report the words per second in the training logs.

    Returns:
      A dictionary mapping a counter name to a Python value.
    """
    counters = {}
    for name, counter in self._words_counters.items():
      counters[name] = self._all_reduce_sum(counter.read_value())
      counter.assign(tf.constant(0, dtype=tf.int64))
    return counters

  def _broadcast_variables(self):
    """Broadcasts variables to other replicas, if required."""
    return

  def _all_reduce_sum(self, value):
    """Reduces the value across all replicas."""
    return value


class HorovodTrainer(Trainer):
  """Trainer compatible with Horovod distributed training."""

  def __init__(self, model, optimizer, hvd, checkpoint=None):
    """Initializes the Horovod trainer.

    Args:
      model: A :class:`opennmt.models.Model` instance to train.
      optimizer: A ``tf.keras.optimizers.Optimizer`` instance.
      hvd: The global Horovod object.
      checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance. If
        not set, no checkpoints will be saved.
    """
    super().__init__(model, optimizer, checkpoint=checkpoint, is_master=hvd.rank() == 0)
    self._hvd = hvd

  @property
  def num_replicas(self):
    return self._hvd.size()

  def _finalize_dataset(self, dataset):
    if callable(dataset):
      dataset = dataset(tf.distribute.InputContext(
          num_input_pipelines=self._hvd.size(),
          input_pipeline_id=self._hvd.rank(),
          num_replicas_in_sync=self._hvd.size()))
    return dataset

  def _broadcast_variables(self):
    self._hvd.broadcast_variables(self._model.variables, root_rank=0)
    self._hvd.broadcast_variables(self._optimizer.variables(), root_rank=0)

  def _all_reduce_sum(self, value):
    return self._hvd.allreduce(value, op=self._hvd.Sum)


class MirroredStrategyTrainer(Trainer):
  """Trainer based on ``tf.distribute.MirroredStrategy`` for local multi-GPU training."""

  def __init__(self, model, optimizer, checkpoint=None, devices=None):
    """Initializes the MirroredStrategy trainer.

    Args:
      model: A :class:`opennmt.models.Model` instance to train.
      optimizer: A ``tf.keras.optimizers.Optimizer`` instance.
      checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance. If
        not set, no checkpoints will be saved.
      devices: List of device strings to use for training. If not set, all
        visible GPUs are used.
    """
    super().__init__(model, optimizer, checkpoint=checkpoint)
    self._strategy = tf.distribute.MirroredStrategy(devices=devices)
    with self._strategy.scope():
      # Create some variables under the strategy scope.
      _ = self._optimizer.iterations

  @property
  def num_replicas(self):
    return self._strategy.num_replicas_in_sync

  def _finalize_dataset(self, dataset):
    # We prefer not to use experimental_distribute_dataset here because it
    # sometimes fails to split the batches (noticed with tokens batch type).
    # We also assume for now that we are training with a single worker
    # otherwise we would need to correctly shard the input dataset.
    dataset_fn = dataset if callable(dataset) else lambda _: dataset
    return self._strategy.experimental_distribute_datasets_from_function(dataset_fn)

  def _forward_and_step(self, source, target, record_summaries=False):
    per_replica_loss = self._strategy.run(
        super()._forward_and_step,
        args=(source, target),
        kwargs=dict(record_summaries=record_summaries))
    return self._strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

  def _forward(self, source, target, record_summaries=False):
    per_replica_loss = self._strategy.run(
        super()._forward,
        args=(source, target),
        kwargs=dict(record_summaries=record_summaries))
    # TODO: this reduction could be delayed until _step is called.
    return self._strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

  def _step(self):
    self._strategy.run(super()._step)


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
    avg = int(counter.numpy() / elapsed_time)
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
  if isinstance(should_record, bool) and not should_record:
    return
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


class _LossScaleOptimizer(tf.keras.mixed_precision.experimental.LossScaleOptimizer):
  # TODO: Remove this wrapper when this fix is released:
  # https://github.com/tensorflow/tensorflow/commit/d1dd08dd2807ac80a4508686618419826463374b

  def get_unscaled_gradients(self, grads):
    loss_scale_reciprocal = 1. / self._loss_scale()
    return [
        _multiply_gradient(g, loss_scale_reciprocal) if g is not None else None
        for g in grads
    ]


def _multiply_gradient(gradient, scale):
  """Multiply a (possibly sparse) gradient by the given scale factor."""
  scale = tf.cast(scale, gradient.dtype)
  if isinstance(gradient, tf.IndexedSlices):
    return tf.IndexedSlices(
        gradient.values * scale,
        gradient.indices,
        dense_shape=gradient.dense_shape)
  else:
    return gradient * scale
