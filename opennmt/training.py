"""Training related classes and functions."""

import os
import time

import tensorflow as tf

from opennmt.optimizers import utils as optimizer_util
from opennmt.utils import misc


class Trainer(object):
  """Model trainer."""

  def __init__(self, checkpoint, devices=None, mixed_precision=False):
    """Initializes the trainer.

    Args:
      checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance.
      devices: List of device strings to use for training.
      mixed_precision: Whether mixed precision is enabled or not.
    """
    if not devices:
      devices = misc.get_devices(count=1)  # Train with 1 device by default.
    self._checkpoint = checkpoint
    self._mixed_precision = mixed_precision
    self._model = checkpoint.model
    self._strategy = tf.distribute.MirroredStrategy(devices=devices)
    self._summary_writer = tf.summary.create_file_writer(checkpoint.model_dir)

    optimizer = checkpoint.optimizer
    if optimizer is None:
      raise ValueError("No optimizer is defined")
    if mixed_precision:
      optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")
    self._optimizer = optimizer

    self._words_counters = {}
    with self._strategy.scope():
      # Create some variables under the strategy scope.
      _ = self._optimizer.iterations
      self._model.create_variables()
      self._gradient_accumulator = optimizer_util.GradientAccumulator()

  def __call__(self,
               dataset,
               max_step=None,
               accum_steps=1,
               report_steps=100,
               save_steps=5000,
               evaluator=None,
               eval_steps=5000,
               export_on_best=None):
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
      export_on_best: Export a SavedModel when this evaluation metric has the
        best value so far.
    """
    if max_step is not None and self._optimizer.iterations.numpy() >= max_step:
      tf.get_logger().warning("Model already reached max_step = %d. Exiting.", max_step)
      return
    if evaluator is not None and evaluator.should_stop():
      tf.get_logger().warning("Early stopping conditions are already met. Exiting.")
      return

    self._gradient_accumulator.reset()
    self._words_counters.clear()

    last_report_time = time.time()
    last_step = 0

    with self._summary_writer.as_default():
      if self._optimizer.iterations.numpy() == 0:
        self._checkpoint.save(0)
      self._model.visualize(self._checkpoint.model_dir)

      for step, loss in self._steps(dataset, accum_steps=accum_steps, report_steps=report_steps):
        last_step = step
        if step % report_steps == 0:
          last_report_time = _report_training_status(
              step,
              loss,
              self._optimizer.learning_rate,
              self._synchronize_words_counters(),
              last_report_time)
        if save_steps is not None and step % save_steps == 0:
          self._checkpoint.save(step)
        if evaluator is not None and eval_steps is not None and step % eval_steps == 0:
          self._evaluate(evaluator, step, export_on_best=export_on_best)
          if evaluator.should_stop():
            tf.get_logger().warning("Early stopping conditions are met. Exiting.")
            break
        if step == max_step:
          break

    if evaluator is not None and last_step != evaluator.last_evaluated_step:
      self._evaluate(evaluator, last_step, export_on_best=export_on_best)
    self._checkpoint.save(last_step)

  def _evaluate(self, evaluator, step, export_on_best=None):
    metrics = evaluator(step)
    if export_on_best is not None and evaluator.is_best(export_on_best):
      export_dir = os.path.join(self._checkpoint.model_dir, "export", str(step))
      tf.get_logger().info("Exporting SavedModel to %s (best %s so far: %f)",
                           export_dir, export_on_best, metrics[export_on_best])
      self._model.export(export_dir)

  def _steps(self, dataset, accum_steps=1, report_steps=None):
    """Returns a generator over training steps."""
    for i, loss in enumerate(self._accumulate_next_gradients(dataset, report_steps=report_steps)):
      if tf.math.is_nan(loss):
        raise RuntimeError("Model diverged with loss = NaN.")
      if i == 0 or (i + 1) % accum_steps == 0:
        self._apply_gradients()
        yield self._optimizer.iterations.numpy(), loss

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
      tf.summary.experimental.set_step(self._optimizer.iterations)
      if report_steps is None:
        should_record_summaries = False
      else:
        should_record_summaries = tf.logical_and(
            tf.equal(self._optimizer.iterations % report_steps, 0),
            tf.equal(self._gradient_accumulator.step, 0))
      with tf.summary.record_if(should_record_summaries):
        per_replica_source, per_replica_target = next(iterator)
        return self._accumulate_gradients(per_replica_source, per_replica_target)

    while True:
      try:
        yield _accumulate_next()
      except tf.errors.OutOfRangeError:
        break

  def _accumulate_gradients(self, per_replica_source, per_replica_target):
    """Accumulates the gradients (cross-replica)."""
    per_replica_loss = self._strategy.experimental_run_v2(
        self._accumulate_gradients_on_replica,
        args=(per_replica_source, per_replica_target))
    # TODO: this reduction could be delayed until _step is called.
    return self._strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_loss, None)

  def _accumulate_gradients_on_replica(self, source, target):
    """Accumulates the gradients (in replica)."""
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
    variables = self._model.trainable_variables
    training_loss = self._model.regularize_loss(training_loss, variables=variables)
    gradients = self._optimizer.get_gradients(training_loss, variables)
    self._gradient_accumulator(gradients)
    tf.summary.scalar("gradients/global_norm", tf.linalg.global_norm(gradients))
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
    counter.assign_add(tf.cast(num_words, tf.int64), read_value=False)

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
      counter.assign(tf.constant(0, dtype=tf.int64), read_value=False)

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


def _report_training_status(step, loss, learning_rate, words_counters, last_report_time):
  tf.summary.experimental.set_step(step)
  new_report_time = time.time()
  words_per_sec_fmt = []
  for name, counter in words_counters.items():
    avg = int(counter.numpy() / (new_report_time - last_report_time))
    tf.summary.scalar(
        "words_per_sec/%s" % name,
        avg,
        description="%s words per second" % name.capitalize())
    fmt = "%s words/s = %d" % (name, avg)
    words_per_sec_fmt.append(fmt)
  words_per_sec_fmt = sorted(words_per_sec_fmt)
  if isinstance(learning_rate, tf.optimizers.schedules.LearningRateSchedule):
    learning_rate = learning_rate(step)
  tf.get_logger().info(
      "Step = %d ; %s ; Learning rate = %f ; Loss = %f",
      step,
      ", ".join(words_per_sec_fmt),
      learning_rate,
      loss)
  tf.summary.scalar("loss", loss, description="Training loss")
  tf.summary.scalar("optim/learning_rate", learning_rate, description="Learning rate")
  return new_report_time
