"""Example of a translation client."""

from __future__ import print_function

import argparse
import threading

import tensorflow as tf

from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def print_batch(batch_tokens):
  """Convenience function to display a batch of tokens."""
  for tokens in batch_tokens:
    print("\t{}".format(" ".join(tokens)))

def pad_batch(batch_tokens):
  """Pads a batch of tokens.

  Args:
    batch_tokens: A list of list of strings.

  Returns:
    batch_tokens: A list of right-padded list of strings.
    lengths: A list of int containing the length of each sequence.
  """
  max_length = 0
  for tokens in batch_tokens:
    max_length = max(max_length, len(tokens))
  lengths = []
  for tokens in batch_tokens:
    length = len(tokens)
    tokens += [""] * (max_length - length)
    lengths.append(length)
  return batch_tokens, lengths

def parse_translation_result(result):
  """Parses the server result.

  Args:
    result: A `PredictResponse` proto.

  Returns:
    A list of list of strings (a.k.a a batch of tokens).
  """
  lengths = tf.make_ndarray(result.outputs["length"])
  batch_predictions = tf.make_ndarray(result.outputs["tokens"])
  best = []
  for hypotheses, length in zip(batch_predictions, lengths):
    # Only consider the first hypothesis (the best one).
    best_hypotheses = hypotheses[0]
    best_length = length[0]
    best.append(best_hypotheses[0:best_length - 1]) # Ignore </s>.
  return best

def done_callback(result_future, condition):
  """Request result callback."""
  exception = result_future.exception()
  if exception:
    print(exception)
  else:
    batch_tokens = parse_translation_result(result_future.result())
    print("Output:")
    print_batch(batch_tokens)

  # Notify the waiting thread that we finished.
  with condition:
    condition.notify()

def send_translation_request(model_name, batch_tokens, stub, condition, timeout=5.0):
  """Sends a translation request.

  Args:
    model_name: The model to request.
    batch_tokens: A list of list of strings.
    stub: The prediction service stub.
    condition: The condition variable to notify.
    timeout: Timeout after this many seconds.
  """
  print("Input:")
  print_batch(batch_tokens)

  batch_tokens, lengths = pad_batch(batch_tokens)
  batch_size = len(batch_tokens)
  max_length = len(batch_tokens[0])

  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = "predictions"

  request.inputs["tokens"].CopyFrom(
      tf.make_tensor_proto(batch_tokens, shape=(batch_size, max_length)))
  request.inputs["length"].CopyFrom(
      tf.make_tensor_proto(lengths, shape=(batch_size,)))

  print("Sending request...")
  result_future = stub.Predict.future(request, timeout)
  result_future.add_done_callback(lambda x: done_callback(x, condition))


def main():
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("--model_name", required=True,
                      help="model name")
  parser.add_argument("--host", default="localhost",
                      help="model server host")
  parser.add_argument("--port", type=int, default=9000,
                      help="model server port")
  parser.add_argument("--timeout", type=float, default=5.0,
                      help="request timeout")
  args = parser.parse_args()

  channel = implementations.insecure_channel(args.host, args.port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  condition = threading.Condition()

  # Prepare a simple batch request with 3 sentences.
  batch_tokens = [
      ["Hello", "world", "!"],
      ["My", "name", "is", "John", "."],
      ["I", "live", "on", "the", "West", "coast", "."]]

  send_translation_request(args.model_name, batch_tokens, stub, condition, timeout=args.timeout)

  # Wait for all request to end.
  with condition:
    condition.wait()

if __name__ == "__main__":
  main()
