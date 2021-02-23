"""Inference related classes and functions."""

import sys
import time

import tensorflow as tf

from opennmt.utils import misc


def predict_dataset(
    model, dataset, print_params=None, predictions_file=None, log_time=False
):
    """Outputs the model predictions for the dataset.

    To run inference on strings directly, see
    :meth:`opennmt.models.Model.serve_function`.

    Args:
      model: A :class:`opennmt.models.Model` instance.
      dataset: A ``tf.data.Dataset`` instance outputting features.
      print_params: A dictionary of parameters passed to
        :meth:`opennmt.models.Model.print_prediction`.
      predictions_file: If set, predictions are saved in this file, otherwise they
        are printed on the standard output.
      log_time: If ``True``, several time metrics will be printed in the logs at
        the end of the inference loop.
    """
    if predictions_file:
        stream = open(predictions_file, encoding="utf-8", mode="w")
    else:
        stream = sys.stdout

    infer_fn = tf.function(model.infer, input_signature=(dataset.element_spec,))
    if not tf.config.functions_run_eagerly():
        tf.get_logger().info("Tracing and optimizing the inference graph...")
        infer_fn.get_concrete_function()  # Trace the function now.

    # Inference might return out-of-order predictions. The OrderRestorer utility is
    # used to write predictions in their original order.
    write_fn = lambda prediction: (
        model.print_prediction(prediction, params=print_params, stream=stream)
    )
    index_fn = lambda prediction: prediction.get("index")
    ordered_writer = misc.OrderRestorer(index_fn, write_fn)

    total_time = 0
    total_tokens = 0
    total_examples = 0
    start_time = time.time()

    # When the inference dataset is bucketized, it can happen that no output is
    # written in a long time. To avoid confusion and give the impression that
    # the process is stuck, we ensure that something is logged regularly.
    max_time_without_output = 10
    last_output_time = start_time

    for features in dataset:
        predictions = infer_fn(features)
        predictions = tf.nest.map_structure(lambda t: t.numpy(), predictions)
        batch_time = time.time()

        for prediction in misc.extract_batches(predictions):
            written = ordered_writer.push(prediction)
            if written:
                last_output_time = batch_time
            else:
                time_without_output = batch_time - last_output_time
                if time_without_output >= max_time_without_output:
                    tf.get_logger().info(
                        "%d predictions are buffered, but waiting for the prediction of "
                        "line %d to advance the output...",
                        ordered_writer.buffer_size,
                        ordered_writer.next_index + 1,
                    )
                    last_output_time = batch_time

        if log_time:
            batch_size = next(iter(predictions.values())).shape[0]
            total_examples += batch_size
            length = predictions.get("length")
            if length is not None:
                if len(length.shape) == 2:
                    length = length[:, 0]
                total_tokens += sum(length)

    if log_time:
        end_time = time.time()
        total_time = end_time - start_time
        tf.get_logger().info("Total prediction time (s): %f", total_time)
        tf.get_logger().info(
            "Average prediction time (s): %f", total_time / total_examples
        )
        if total_tokens > 0:
            tf.get_logger().info("Tokens per second: %f", total_tokens / total_time)
    if predictions_file:
        stream.close()


def score_dataset(model, dataset, print_params=None, output_file=None):
    """Outputs the model scores for the dataset.

    Args:
      model: A :class:`opennmt.models.Model` instance.
      dataset: A ``tf.data.Dataset`` instance outputting parallel features and
        labels.
      print_params: A dictionary of parameters passed to
        :meth:`opennmt.models.Model.print_score`.
      output_file: If set, outputs are saved in this file, otherwise they are
        printed on the standard output.
    """
    if output_file:
        stream = open(output_file, encoding="utf-8", mode="w")
    else:
        stream = sys.stdout

    write_fn = lambda batch: (
        model.print_score(batch, params=print_params, stream=stream)
    )
    index_fn = lambda batch: batch.get("index")
    ordered_writer = misc.OrderRestorer(index_fn, write_fn)

    score_fn = tf.function(model.score, input_signature=dataset.element_spec)
    for features, labels in dataset:
        results = score_fn(features, labels)
        results = tf.nest.map_structure(lambda t: t.numpy(), results)
        for batch in misc.extract_batches(results):
            ordered_writer.push(batch)

    if output_file:
        stream.close()
