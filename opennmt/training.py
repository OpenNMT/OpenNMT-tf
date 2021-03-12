"""Training related classes and functions."""

import collections
import contextlib
import itertools
import time

import tensorflow as tf

from opennmt.inputters import text_inputter
from opennmt.optimizers import utils as optimizer_util
from opennmt.utils import compat
from opennmt.utils import misc


def _add_mixed_precision_wrapper(optimizer):
    # TODO: clean mixed precision API when TensorFlow requirement is updated to >=2.4.
    wrapper_class = None
    wrapper_kwargs = {}
    if compat.tf_supports("keras.mixed_precision.LossScaleOptimizer"):
        wrapper_class = tf.keras.mixed_precision.LossScaleOptimizer
    else:
        wrapper_class = tf.keras.mixed_precision.experimental.LossScaleOptimizer
        wrapper_kwargs = dict(loss_scale="dynamic")
    if not isinstance(optimizer, wrapper_class):
        optimizer = wrapper_class(optimizer, **wrapper_kwargs)
    return optimizer


class Trainer:
    """Base class for model trainer, implementing single-GPU training."""

    def __init__(self, model, optimizer, checkpoint=None):
        """Initializes the trainer.

        Args:
          model: A :class:`opennmt.models.Model` instance to train.
          optimizer: A ``tf.keras.optimizers.Optimizer`` instance.
          checkpoint: A :class:`opennmt.utils.checkpoint.Checkpoint` instance. If
            not set, no checkpoints will be saved.
        """
        self._checkpoint = checkpoint
        self._model = model
        if checkpoint is not None:
            self._summary_writer = tf.summary.create_file_writer(checkpoint.model_dir)
        else:
            self._summary_writer = tf.summary.create_noop_writer()
        self._training_stats = None
        self._gradient_accumulator = optimizer_util.GradientAccumulator()
        self._mixed_precision = misc.mixed_precision_enabled()

        if optimizer is None:
            raise ValueError("No optimizer is defined")
        if self._mixed_precision:
            optimizer = _add_mixed_precision_wrapper(optimizer)
        self._optimizer = optimizer

    @property
    def is_master(self):
        """Master replica."""
        return True

    @property
    def num_replicas(self):
        """Number of synchronous training replicas."""
        return 1

    def __call__(
        self,
        dataset,
        max_step=None,
        accum_steps=1,
        report_steps=100,
        save_steps=5000,
        evaluator=None,
        eval_steps=5000,
        moving_average_decay=None,
    ):
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

        Returns:
          A dictionary with various training statistics.
        """
        if max_step is not None and self._optimizer.iterations.numpy() >= max_step:
            raise RuntimeError(
                "The training already reached max_step (%d). If you "
                "want to continue the training, you should increase the "
                "max_step value in the training parameters." % max_step
            )
        if evaluator is not None and evaluator.should_stop():
            raise RuntimeError(
                "The early stopping conditions are already met. If you "
                "want to continue the training, you should update your "
                "early stopping parameters."
            )

        self._gradient_accumulator.reset()

        with self._summary_writer.as_default():
            self._training_stats = TrainingStats(
                self._model, self._optimizer, reduce_fn=self._all_reduce_sum
            )
            iterations = self._optimizer.iterations
            tf.summary.experimental.set_step(iterations)

            step = None
            moving_average = None
            for loss in self._steps(
                dataset, accum_steps=accum_steps, report_steps=report_steps
            ):
                if moving_average_decay is not None and self.is_master:
                    if moving_average is None:
                        moving_average = MovingAverage(
                            self._model.trainable_variables,
                            iterations,
                            decay=moving_average_decay,
                        )
                    self._update_moving_average(moving_average)

                step = iterations.numpy()
                reset_throughput = False
                self._training_stats.update_on_step(step, loss)
                if step % report_steps == 0:
                    self._training_stats.log(self.is_master)
                    reset_throughput = True
                if step == 1 or (save_steps is not None and step % save_steps == 0):
                    self._save_checkpoint(step, moving_average=moving_average)
                    reset_throughput = True
                if eval_steps is not None and step % eval_steps == 0:
                    early_stop = self._evaluate(
                        evaluator, step, moving_average=moving_average
                    )
                    reset_throughput = True
                    if early_stop:
                        tf.get_logger().warning(
                            "Early stopping conditions are met. Exiting."
                        )
                        break
                if step == max_step:
                    break
                if reset_throughput:
                    self._training_stats.reset_throughput()

            if step is None:
                raise RuntimeError(
                    "No training steps were executed. This usually means the "
                    "training file is empty or all examples were filtered out. "
                    "For the latter, verify that the maximum_*_length values are "
                    "consistent with your data."
                )

            self._training_stats.log_final(self.is_master)
            summary = self._training_stats.get_global_summary()
            self._save_checkpoint(step, moving_average=moving_average)
            self._evaluate(evaluator, step, moving_average=moving_average)
            return summary

    def _save_checkpoint(self, step, moving_average=None):
        """Saves a checkpoint for step."""
        if (
            not self.is_master
            or self._checkpoint is None
            or step == self._checkpoint.last_saved_step
        ):
            return
        shadow_variables = (
            moving_average.shadow_variables()
            if moving_average is not None
            else contextlib.suppress()
        )
        with shadow_variables:
            self._checkpoint.save(step)

    def _evaluate(self, evaluator, step, moving_average=None):
        """Runs evaluation for step. Returns ``True`` is early conditions are met."""
        if (
            not self.is_master
            or evaluator is None
            or step == evaluator.last_evaluated_step
        ):
            return False
        shadow_variables = (
            moving_average.shadow_variables()
            if moving_average is not None
            else contextlib.suppress()
        )
        with shadow_variables:
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

        # We define 2 separate functions to support gradient accumulation:
        #  * forward: compute and accumulate the gradients
        #  * step: apply the gradients
        # When gradient accumulation is disabled, the forward function also applies the gradients.

        def _forward():
            # We get the next dataset element within the function for increased efficiency
            # and avoid dealing with tf.function input signatures.
            source, target = next(iterator)
            return self._forward(
                source,
                target,
                accum_steps=accum_steps,
                report_steps=report_steps,
            )

        def _step():
            return self._step()

        # Wrap forward and step with tf.function to run in graph mode.
        forward_fn = tf.function(_forward)
        step_fn = tf.function(_step) if accum_steps > 1 else lambda: None
        step_loss = 0

        for i in itertools.count():
            try:
                loss = forward_fn()
            except (
                StopIteration,
                tf.errors.OutOfRangeError,
            ):  # Dataset iterator exhausted.
                break

            if tf.math.is_nan(loss):
                raise RuntimeError("Model diverged with loss = NaN.")

            step_loss += float(loss)
            if (i + 1) % accum_steps == 0:
                step_fn()
                if i + 1 == accum_steps:
                    self._broadcast_variables()
                yield step_loss
                step_loss = 0

    def _run_model(self, source, target, accum_steps=1):
        """Computes the loss of the given source and target pair.

        Args:
          source: A nested structure of tensors.
          target: A nested structure of tensors.
          accum_steps: The number of gradient accumulation steps.

        Returns:
          A tuple containing,

          - The loss to compute the gradients.
          - The loss to report.
        """
        first_call = not self._model.built
        outputs, _ = self._model(
            source, labels=target, training=True, step=self._optimizer.iterations
        )
        loss = self._model.compute_loss(outputs, target, training=True)
        if isinstance(loss, tuple):
            training_loss = loss[0] / loss[1]
            reported_loss = loss[0] / loss[2] if len(loss) > 2 else training_loss
        else:
            training_loss, reported_loss = loss, loss
        training_loss = self._model.regularize_loss(
            training_loss, variables=self._model.trainable_variables
        )
        loss_scale = accum_steps * self.num_replicas
        training_loss /= loss_scale
        reported_loss /= loss_scale
        self._training_stats.update_on_example(source, target)
        if first_call and self.is_master:
            if self._checkpoint is not None:
                self._model.visualize(self._checkpoint.model_dir)
            tf.get_logger().info(
                "Number of model parameters: %d", self._model.count_params()
            )
            tf.get_logger().info(
                "Number of model weights: %d (trainable = %d, non trainable = %d)",
                len(self._model.weights),
                len(self._model.trainable_weights),
                len(self._model.non_trainable_weights),
            )
        return training_loss, reported_loss

    def _should_record_summaries(self, accum_steps, report_steps):
        """Returns a boolean tensor to be used in tf.summary.record_if."""
        if report_steps is None or not self.is_master:
            return False
        record_summaries = tf.equal(self._optimizer.iterations % report_steps, 0)
        if accum_steps > 1:
            record_summaries = tf.logical_and(
                record_summaries, tf.equal(self._gradient_accumulator.step, 0)
            )
        return record_summaries

    def _compute_gradients(self, source, target, accum_steps, report_steps):
        """Computes the gradient of a training example."""
        record_summaries = self._should_record_summaries(accum_steps, report_steps)
        with tf.summary.record_if(record_summaries):
            if tf.executing_eagerly():
                with tf.GradientTape() as tape:
                    training_loss, reported_loss = self._run_model(
                        source, target, accum_steps=accum_steps
                    )
                    if self._mixed_precision:
                        training_loss = self._optimizer.get_scaled_loss(training_loss)
                gradients = tape.gradient(
                    training_loss, self._model.trainable_variables
                )
                if self._mixed_precision:
                    gradients = self._optimizer.get_unscaled_gradients(gradients)
            else:
                training_loss, reported_loss = self._run_model(
                    source, target, accum_steps=accum_steps
                )
                # In mixed precision training, LossScaleOptimizer.get_gradients takes care
                # of loss scaling.
                gradients = self._optimizer.get_gradients(
                    training_loss, self._model.trainable_variables
                )
            _summarize_gradients(gradients, record_summaries)
        return reported_loss, gradients

    def _apply_gradients(self, gradients):
        """Applies the gradients."""
        self._optimizer.apply_gradients(
            list(zip(gradients, self._model.trainable_variables))
        )

    def _forward(self, source, target, accum_steps=1, report_steps=None):
        """Forwards a training example and accumulates the gradients."""
        loss, gradients = self._compute_gradients(
            source,
            target,
            accum_steps,
            report_steps,
        )
        if accum_steps > 1:
            self._gradient_accumulator(gradients)
        else:
            self._apply_gradients(gradients)
        return loss

    def _step(self):
        """Applies gradients and resets accumulation."""
        self._apply_gradients(self._gradient_accumulator.gradients)
        self._gradient_accumulator.reset()

    def _update_moving_average(self, moving_average):
        """Updates the moving average of variables."""
        moving_average.update()

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
        super().__init__(model, optimizer, checkpoint=checkpoint)
        self._hvd = hvd

    @property
    def is_master(self):
        return self._hvd.rank() == 0

    @property
    def num_replicas(self):
        return self._hvd.size()

    def _finalize_dataset(self, dataset):
        if callable(dataset):
            dataset = dataset(
                tf.distribute.InputContext(
                    num_input_pipelines=self._hvd.size(),
                    input_pipeline_id=self._hvd.rank(),
                    num_replicas_in_sync=self._hvd.size(),
                )
            )
        return dataset

    def _apply_gradients(self, gradients):
        return super()._apply_gradients(map(self._all_reduce_sum, gradients))

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
        dataset_fn = dataset if callable(dataset) else lambda _: dataset
        # TODO: clean this API usage when TensorFlow requirement is updated to >=2.4.
        distribute_fn = getattr(
            self._strategy, "distribute_datasets_from_function", None
        )
        if distribute_fn is None:
            distribute_fn = (
                self._strategy.experimental_distribute_datasets_from_function
            )
        return distribute_fn(dataset_fn)

    def _forward(self, source, target, accum_steps=1, report_steps=None):
        per_replica_loss = self._strategy.run(
            super()._forward,
            args=(source, target),
            kwargs=dict(accum_steps=accum_steps, report_steps=report_steps),
        )
        # TODO: this reduction could be delayed until _step is called.
        return self._strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_loss, None)

    def _step(self):
        self._strategy.run(super()._step)

    def _update_moving_average(self, moving_average):
        with self._strategy.scope():
            super()._update_moving_average(moving_average)


def _summarize_gradients(gradients, should_record):
    # Only compute the gradients global norm when the value is actually recorded.
    if isinstance(should_record, bool) and not should_record:
        return
    tf.summary.scalar(
        "gradients/global_norm",
        tf.cond(
            should_record,
            true_fn=lambda: tf.linalg.global_norm(gradients),
            false_fn=lambda: tf.constant(0, dtype=gradients[0].dtype),
        ),
    )


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
            tf.get_logger().warning(
                "Moving average decay should be close to 1 (e.g. 0.9999) but you "
                "passed %f, is it correct? See https://www.tensorflow.org/api_docs"
                "/python/tf/train/ExponentialMovingAverage for details about the "
                "formula and recommended decay values."
            )
        self._ema = tf.train.ExponentialMovingAverage(decay, num_updates=step)
        self._variables = variables

    @tf.function
    def update(self):
        """Updates the moving average of the variables."""
        self._ema.apply(self._variables)

    @contextlib.contextmanager
    def shadow_variables(self):
        """Returns a context manager that assigns the variables to their moving
        average value on enter and restores the previous value on exit.

        Returns:
          A context manager.
        """
        previous_values = []
        for variable in self._variables:
            previous_values.append(variable.value())
            variable.assign(self._ema.average(variable))
        yield
        for previous_value, variable in zip(previous_values, self._variables):
            variable.assign(previous_value)


class TrainingStats:
    """Aggregate and summarize training statistics."""

    def __init__(self, model, optimizer, reduce_fn=None, warmup_steps=2):
        """Initializes the statistics.

        Args:
          model: The model.
          optimizer: The optimizer.
          reduce_fn: In case of distributed training, a function to sum reduce
            distributed values.
          warmup_steps: Throughput values are ignored for this many steps at the
            start of the training.
        """
        self._model = model
        self._optimizer = optimizer
        self._reduce_fn = reduce_fn
        self._warmup_steps = warmup_steps
        self._words_counters = {}
        self._num_updates = 0
        self._average_loss = 0
        self._last_loss = None
        self._last_step = optimizer.iterations.numpy()
        self._last_logged_step = self._last_step
        self._last_logged_time = time.time()
        self._num_tokens = collections.defaultdict(int)
        self._oov_tokens = collections.defaultdict(lambda: collections.defaultdict(int))

    def update_on_example(self, source, target):
        """Updates the training statistics on a new training example.

        This may be called within a tf.function.

        Args:
          source: A dictionary of source features.
          target: A dictionary of target features.
        """
        self._update_words_counter("source", source)
        self._record_oov_tokens("source", source, self._model.features_inputter)
        if not self._model.unsupervised:
            self._update_words_counter("target", target)
            self._record_oov_tokens("target", target, self._model.labels_inputter)

    def update_on_step(self, step, loss):
        """Updates the training statistics on a new training step.

        Args:
          step: The current training step.
          loss: The loss for this step.
        """
        # Convert Numpy or Tensor values to Python.
        step = int(step)
        loss = float(loss)

        self._last_step = step
        self._last_loss = loss
        self._average_loss = (self._average_loss * self._num_updates + loss) / (
            self._num_updates + 1
        )

        if self._num_updates < self._warmup_steps:
            self.reset_throughput()
        self._num_updates += 1

    def get_last_summary(self):
        """Returns a summary of the training since the last log.

        Returns:
          A dictionary containing various statistics.
        """
        elapsed_time = time.time() - self._last_logged_time
        return {
            "learning_rate": self._get_learning_rate(),
            "loss": self._last_loss,
            "step": self._last_step,
            "steps_per_sec": (self._last_step - self._last_logged_step) / elapsed_time,
            "words_per_sec": {
                name: int(value.numpy() / elapsed_time)
                for name, value in self._get_words_counters().items()
            },
        }

    def get_global_summary(self):
        """Returns a summary of the training since the beginning.

        Returns:
          A dictionary containing various statistics.
        """
        return {
            "average_loss": self._average_loss,
            "last_learning_rate": self._get_learning_rate(),
            "last_loss": self._last_loss,
            "last_step": self._last_step,
            "num_steps": self._num_updates,
        }

    def log(self, is_master=True):
        """Logs the last training statistics.

        Args:
          is_master: Whether this process is the master worker or not.
        """

        # Only the master should log the training statistics but we build the
        # summary on all workers since it may reduce distributed values.
        summary = self.get_last_summary()

        if not is_master:
            return

        tf.summary.scalar(
            "steps_per_sec",
            summary["steps_per_sec"],
            description="Training steps per second",
        )
        steps_per_sec_fmt = "steps/s = %0.2f" % summary["steps_per_sec"]

        words_per_sec_fmt = []
        for name, avg in summary["words_per_sec"].items():
            tf.summary.scalar(
                "words_per_sec/%s" % name,
                avg,
                description="%s words per second" % name.capitalize(),
            )
            words_per_sec_fmt.append("%s words/s = %d" % (name, avg))

        tf.get_logger().info(
            "Step = %d ; %s ; Learning rate = %f ; Loss = %f",
            summary["step"],
            ", ".join([steps_per_sec_fmt] + list(sorted(words_per_sec_fmt))),
            summary["learning_rate"],
            summary["loss"],
        )
        tf.summary.scalar("loss", summary["loss"], description="Training loss")
        tf.summary.scalar(
            "optim/learning_rate", summary["learning_rate"], description="Learning rate"
        )

    def log_final(self, is_master=True):
        """Outputs the final log."""
        if not is_master:
            return

        for name, oov_tokens in self._oov_tokens.items():
            num_oov_tokens = sum(oov_tokens.values())
            if num_oov_tokens > 0:
                num_tokens = self._num_tokens[name]
                tf.get_logger().warning(
                    "%.3f%% of %s tokens are out of vocabulary (%d out of %d tokens)",
                    (num_oov_tokens / num_tokens) * 100,
                    name,
                    num_oov_tokens,
                    num_tokens,
                )
                most_frequent_oov_tokens = (
                    "%s (%.1f%%)" % (oov_token, (count / num_oov_tokens) * 100)
                    for oov_token, count in sorted(
                        oov_tokens.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                )
                most_frequent_oov_tokens = list(most_frequent_oov_tokens)[:10]
                tf.get_logger().info(
                    "The %d most frequent out of vocabulary %s tokens are: %s",
                    len(most_frequent_oov_tokens),
                    name,
                    "; ".join(most_frequent_oov_tokens),
                )

    def reset_throughput(self):
        """Resets the accumulated values since the last log."""
        self._reset_words_counters()
        self._last_logged_step = self._last_step
        self._last_logged_time = time.time()

    def _get_learning_rate(self):
        learning_rate = self._optimizer.learning_rate
        if isinstance(learning_rate, tf.optimizers.schedules.LearningRateSchedule):
            learning_rate = learning_rate(self._last_step)
        return float(learning_rate)

    def _record_oov_tokens(self, name, features, inputter):
        if not isinstance(inputter, text_inputter.WordEmbedder):
            return

        def _record(num_tokens, oov_tokens):
            self._num_tokens[name] += int(num_tokens)
            all_oov_tokens = self._oov_tokens[name]
            for oov_token in oov_tokens.flatten():
                all_oov_tokens[oov_token.decode("utf-8")] += 1

        num_tokens = tf.reduce_sum(
            inputter.get_length(features, ignore_special_tokens=True)
        )
        oov_tokens = inputter.get_oov_tokens(features)

        tf.numpy_function(_record, [num_tokens, oov_tokens], [])

    def _update_words_counter(self, name, features):
        """Accumulates the number of source and target tokens to report throughput."""
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
                aggregation=tf.VariableAggregation.SUM,
            )
            self._words_counters[name] = counter
        counter.assign_add(tf.cast(num_words, tf.int64))

    @tf.function
    def _get_words_counters(self):
        """Returns the accumulated words counters.

        Returns:
          A dictionary mapping a counter name to a value.
        """
        counters = {}
        for name, counter in self._words_counters.items():
            counter = counter.read_value()
            if self._reduce_fn is not None:
                counter = self._reduce_fn(counter)
            counters[name] = counter
        return counters

    @tf.function
    def _reset_words_counters(self):
        """Resets the accumulated words counters."""
        for counter in self._words_counters.values():
            counter.assign(tf.constant(0, dtype=tf.int64))
