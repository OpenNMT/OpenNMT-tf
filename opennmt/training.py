"""Training related classes and functions."""

import collections
import itertools
import tempfile
import time
import six

import tensorflow as tf


class Trainer(object):
  """Model trainer."""

  def __init__(self, model, optimizer, params=None, model_dir=None, keep_checkpoint_max=8):
    """Initializes the trainer.

    Args:
      model: A :class:`opennmt.models.model.Model` to train.
      optimizer: The optimizer instance.
      params: A dictionary of hyperparameters.
      model_dir: The directory where checkpoints will be saved. If not set, a
        temporary directory will be used.
      keep_checkpoint_max: The maximum number of checkpoints to keep.
    """
    if params is None:
      params = {}
    if model_dir is None:
      model_dir = tempfile.mkdtemp()
    self._model = model
    self._optimizer = optimizer
    self._params = params
    self._model_dir = model_dir
    self._checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    self._checkpoint_manager = tf.train.CheckpointManager(
        self._checkpoint, model_dir, keep_checkpoint_max)

  @classmethod
  def from_config(cls, model, config):
    """Creates a trainer from the user configuration.

    Args:
      model: A :class:`opennmt.models.model.Model` to train.
      config: The global user configuration.

    Returns:
      A :class:`opennmt.training.Trainer` instance.
    """
    params = config["params"]
    train_config = config["train"]
    optimizer = model.get_optimizer(params=params)
    return cls(
        model,
        optimizer,
        params=params,
        model_dir=config.get("model_dir"),
        keep_checkpoint_max=train_config.get("keep_checkpoint_max", 8))

  def save_checkpoint(self, step):
    """Saves a checkpoint for :obj:`step`."""
    path = self._checkpoint_manager.save(checkpoint_number=step)
    tf.get_logger().info("Saved checkpoint %s", path)

  def restore_checkpoint(self, checkpoint_path=None):
    """Restores a checkpoint.

    When :obj:`checkpoint_path` is not set, the training will continue from the
    last checkpoint of ``model_dir``. Otherwise, model weights will be loaded
    from :obj:`checkpoint_path`.

    Args:
      checkpoint_path: Path to a checkpoint or a directory containing
        checkpoints.

    Returns:
      Path to the checkpoint that was restored.
    """
    if checkpoint_path is not None:
      checkpoint = tf.train.Checkpoint(model=self._model)
      if tf.io.gfile.isdir(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
      checkpoint = self._checkpoint
      if self._checkpoint_manager.latest_checkpoint is not None:
        checkpoint_path = self._checkpoint_manager.latest_checkpoint
    if checkpoint_path is not None:
      checkpoint.restore(checkpoint_path)
      tf.get_logger().info("Restored checkpoint %s", checkpoint_path)
    return checkpoint_path

  def __call__(self,
               dataset,
               max_step=None,
               accum_steps=1,
               report_steps=100,
               save_steps=5000,
               evaluator=None,
               eval_steps=5000):
    """Runs the training.

    Args:
      dataset: A training dataset.
      max_step: The final training step.
      accum_steps: The number of gradient accumulation steps.
      report_steps: Report status every this many steps.
      save_steps: Save a checkpoint every this many steps.
      evaluator: A :class:`opennmt.evaluation.Evaluator` instance to call for
        evaluation.
      eval_steps: Evaluate every this many steps.
    """
    if max_step is not None and self._optimizer.iterations.numpy() >= max_step:
      tf.get_logger().warn("Model already reached train_steps = %d. Exiting.", max_step)
      return

    iterator = iter(dataset)
    gradients = []
    variables = []

    @tf.function
    def _step():
      source, target = next(iterator)
      outputs, _ = self._model(source, target, self._params, tf.estimator.ModeKeys.TRAIN)
      loss = self._model.compute_loss(outputs, target, training=True, params=self._params)
      loss = loss[0] / loss[1]

      if not variables:
        trainable_variables = self._model.trainable_variables
        freeze_variables = self._params.get("freeze_variables")
        if freeze_variables:
          trainable_variables = _get_trainable_variables(trainable_variables, freeze_variables)
        variables.extend(trainable_variables)

      step_gradients = self._optimizer.get_gradients(loss, variables)
      if not gradients:
        for step_gradient in step_gradients:
          gradients.append(tf.Variable(tf.zeros_like(step_gradient), trainable=False))
      for gradient, step_gradient in zip(gradients, step_gradients):
        gradient.assign_add(step_gradient)

      num_words = {}
      if "length" in source:
        num_words["source"] = tf.reduce_sum(source["length"])
      if "length" in target:
        num_words["target"] = tf.reduce_sum(target["length"])
      return loss, num_words

    @tf.function
    def _apply_gradients():
      self._optimizer.apply_gradients(list(zip(gradients, variables)))
      for gradient in gradients:
        gradient.assign(tf.zeros_like(gradient))

    accum_num_words = collections.defaultdict(int)
    last_report_time = time.time()
    last_step = 0

    for i in itertools.count():
      try:
        loss, num_words = _step()
      except tf.errors.OutOfRangeError:
        break

      if i == 0 or (i + 1) % accum_steps == 0:
        _apply_gradients()

      for key, value in six.iteritems(num_words):
        accum_num_words[key] += value.numpy()
      step = self._optimizer.iterations.numpy()
      if step == last_step:
        continue  # Do not process same step twice.
      last_step = step
      if step % report_steps == 0:
        last_report_time = _report_training_status(
            step,
            loss,
            self._optimizer.learning_rate,
            accum_num_words,
            last_report_time)
      if save_steps is not None and step % save_steps == 0:
        self.save_checkpoint(step)
      if evaluator is not None and eval_steps is not None and step % eval_steps == 0:
        evaluator(step)
      if step == max_step:
        break

    self.save_checkpoint(step)


def _report_training_status(step, loss, learning_rate, accum_num_words, last_report_time):
  new_report_time = time.time()
  words_per_sec_fmt = []
  for key, value in six.iteritems(accum_num_words):
    avg = value / (new_report_time - last_report_time)
    accum_num_words[key] = 0
    fmt = "%s words/s = %d" % (key, int(avg))
    words_per_sec_fmt.append(fmt)
  if isinstance(learning_rate, tf.optimizers.schedules.LearningRateSchedule):
    learning_rate = learning_rate(step)
  tf.get_logger().info(
      "Step = %d ; %s ; Learning rate = %f ; Loss = %f",
      step,
      ", ".join(words_per_sec_fmt),
      learning_rate,
      loss)
  return new_report_time

def _print_variables(variables, name=None):
  if name is not None:
    tf.get_logger().info("%s variables:", name.capitalize())
  for variable in variables:
    tf.get_logger().info(" * %s", variable.name)

def _get_trainable_variables(variables, freeze_variables):
  if not isinstance(freeze_variables, list):
    freeze_variables = [freeze_variables]
  regexs = list(map(re.compile, freeze_variables))
  frozen_variables = []
  trainable_variables = []
  for variable in variables:
    if any(regex.match(variable.name) for regex in regexs):
      frozen_variables.append(variable)
    else:
      trainable_variables.append(variable)
  _print_variables(frozen_variables, name="frozen")
  _print_variables(trainable_variables, name="trainable")
  return trainable_variables
