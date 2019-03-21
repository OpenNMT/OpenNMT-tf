"""Functions for Estimator API integration."""

import copy

import tensorflow as tf

from opennmt.utils import hooks


def make_serving_input_fn(model):
  """Returns the serving input function.

  Args:
    model: An initialized :class:`opennmt.models.model.Model` instance.

  Returns:
    A callable that returns a ``tf.estimator.export.ServingInputReceiver``.
  """

  def _fn():
    local_model = copy.deepcopy(model)
    input_signature = local_model.features_inputter.input_signature()
    if input_signature is None:
      raise NotImplementedError("This model does not define an input signature.")
    receiver_tensors = tf.nest.map_structure(
        lambda spec: tf.compat.v1.placeholder(spec.dtype, shape=spec.shape), input_signature)
    features = local_model.features_inputter.make_features(features=receiver_tensors.copy())
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  return _fn

def make_input_fn(model,
                  mode,
                  batch_size,
                  features_file,
                  labels_file=None,
                  batch_type="examples",
                  batch_multiplier=1,
                  bucket_width=None,
                  maximum_features_length=None,
                  maximum_labels_length=None,
                  shuffle_buffer_size=None,
                  single_pass=False,
                  num_shards=1,
                  shard_index=0,
                  num_threads=None,
                  prefetch_buffer_size=None):
  """Creates the input function.

  Args:
    model: An initialized :class:`opennmt.models.model.Model` instance.
    mode: A ``tf.estimator.ModeKeys`` mode.
    batch_size: The batch size to use.
    features_file: The file containing input features.
    labels_file: The file containing output labels.
    batch_type: The training batching stragety to use: can be "examples" or
      "tokens".
    batch_multiplier: The batch size multiplier to prepare splitting accross
       replicated graph parts.
    bucket_width: The width of the length buckets to select batch candidates
      from. ``None`` to not constrain batch formation.
    maximum_features_length: The maximum length or list of maximum lengths of
      the features sequence(s). ``None`` to not constrain the length.
    maximum_labels_length: The maximum length of the labels sequence.
      ``None`` to not constrain the length.
    shuffle_buffer_size: The number of elements from which to sample.
    single_pass: If ``True``, makes a single pass over the training data.
    num_shards: The number of data shards (usually the number of workers in a
      distributed setting).
    shard_index: The shard index this input pipeline should read from.
    num_threads: The number of elements processed in parallel.
    prefetch_buffer_size: The number of batches to prefetch asynchronously. If
      ``None``, use an automatically tuned value on TensorFlow 1.8+ and 1 on
      older versions.

  Returns:
    The input function.

  See Also:
    ``tf.estimator.Estimator``.
  """
  batch_size_multiple = 1
  if batch_type == "tokens" and model.dtype == tf.float16:
    batch_size_multiple = 8

  def _fn():
    local_model = copy.deepcopy(model)

    if mode == tf.estimator.ModeKeys.PREDICT:
      dataset = local_model.examples_inputter.make_inference_dataset(
          features_file,
          batch_size,
          bucket_width=bucket_width,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)
    elif mode == tf.estimator.ModeKeys.EVAL:
      dataset = local_model.examples_inputter.make_evaluation_dataset(
          features_file,
          labels_file,
          batch_size,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)
    elif mode == tf.estimator.ModeKeys.TRAIN:
      dataset = local_model.examples_inputter.make_training_dataset(
          features_file,
          labels_file,
          batch_size,
          batch_type=batch_type,
          batch_multiplier=batch_multiplier,
          batch_size_multiple=batch_size_multiple,
          shuffle_buffer_size=shuffle_buffer_size,
          bucket_width=bucket_width,
          maximum_features_length=maximum_features_length,
          maximum_labels_length=maximum_labels_length,
          single_pass=single_pass,
          num_shards=num_shards,
          shard_index=shard_index,
          num_threads=num_threads,
          prefetch_buffer_size=prefetch_buffer_size)

    return dataset

  return _fn

def make_model_fn(model,
                  eval_prediction_hooks_fn=None,
                  hvd=None):
  """Creates the model function.

  Args:
    model: An initialized but not built :class:`opennmt.models.model.Model`
      instance.
    eval_prediction_hooks_fn: A callable that takes the model predictions
      during evaluation and return an iterable of evaluation hooks (e.g. for
      saving predictions on disk, running external evaluators, etc.).
    hvd: Optional Horovod object.

  See Also:
    ``tf.estimator.Estimator`` 's ``model_fn`` argument for more details about
    arguments and the returned value.
  """

  def _fn(features, labels, params, mode, config):
    """model_fn implementation."""
    eval_metric_ops = None
    evaluation_hooks = None
    export_outputs = None
    loss = None
    train_op = None
    training_hooks = None

    step = tf.compat.v1.train.get_or_create_global_step()
    local_model = copy.deepcopy(model)
    optimizer = local_model.get_optimizer(step, params=params)
    checkpoint = tf.train.Checkpoint(
        step=step,
        model=local_model,
        optimizer=optimizer)

    outputs, predictions = local_model(features, labels, params, mode)
    if labels is not None:
      loss = local_model.compute_loss(
          outputs,
          labels,
          training=mode == tf.estimator.ModeKeys.TRAIN,
          params=params)
      loss = _extract_loss(loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
      variables = local_model.trainable_variables
      gradients = optimizer.get_gradients(loss, variables)
      train_op = tf.group(
          optimizer.apply_gradients(zip(gradients, variables)),
          step.assign_add(1))
      training_hooks = []
      if config is not None:
        local_model.examples_inputter.visualize(config.model_dir)
        features_length = local_model.features_inputter.get_length(features)
        labels_length = (
            local_model.labels_inputter.get_length(labels)
            if not local_model.unsupervised else None)
        num_words = {}
        if features_length is not None:
          num_words["source"] = tf.reduce_sum(features_length)
        if labels_length is not None:
          num_words["target"] = tf.reduce_sum(labels_length)
        training_hooks.append(hooks.LogWordsPerSecondHook(
            num_words,
            step,
            every_n_steps=config.save_summary_steps,
            output_dir=config.model_dir))

    elif mode == tf.estimator.ModeKeys.EVAL:
      eval_metric_ops = local_model.compute_metrics(predictions, labels)
      evaluation_hooks = []
      if predictions is not None and eval_prediction_hooks_fn is not None:
        evaluation_hooks.extend(eval_prediction_hooks_fn(predictions, step))

    elif mode == tf.estimator.ModeKeys.PREDICT:
      # Forward example index for reordering predictions.
      if "index" in features:
        predictions["index"] = features["index"]
      export_outputs = {}
      export_outputs[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
          tf.estimator.export.PredictOutput(predictions))

    return tf.estimator.EstimatorSpec(
        mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops,
        export_outputs=export_outputs,
        training_hooks=training_hooks,
        scaffold=tf.compat.v1.train.Scaffold(saver=checkpoint),
        evaluation_hooks=evaluation_hooks)

  return _fn


def _normalize_loss(num, den=None):
  """Normalizes the loss."""
  if den is not None:
    return num / den
  else:
    return num

def _extract_loss(loss):
  """Extracts and summarizes the loss."""
  if not isinstance(loss, tuple):
    actual_loss = loss
    tboard_loss = actual_loss
  else:
    actual_loss = _normalize_loss(loss[0], den=loss[1])
    tboard_loss = _normalize_loss(loss[0], den=loss[2]) if len(loss) > 2 else actual_loss
  tf.summary.scalar("loss", tboard_loss)
  return actual_loss
