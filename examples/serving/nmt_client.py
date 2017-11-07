"""Example of a translation client."""

from __future__ import print_function

import argparse

import tensorflow as tf

from grpc.beta import implementations

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


def parse_translation_result(result):
  """Parses a translation result.

  Args:
    result: A `PredictResponse` proto.

  Returns:
    A list of tokens.
  """
  lengths = tf.make_ndarray(result.outputs["length"])[0]
  hypotheses = tf.make_ndarray(result.outputs["tokens"])[0]

  # Only consider the first hypothesis (the best one).
  best_hypothesis = hypotheses[0]
  best_length = lengths[0]

  return best_hypothesis[0:best_length - 1] # Ignore </s>

def translate(stub, model_name, tokens, timeout=5.0):
  """Translates a sequence of tokens.

  Args:
    stub: The prediction service stub.
    model_name: The model to request.
    tokens: A list of tokens.
    timeout: Timeout after this many seconds.

  Returns:
    A future.
  """
  length = len(tokens)

  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.inputs["tokens"].CopyFrom(
      tf.make_tensor_proto([tokens], shape=(1, length)))
  request.inputs["length"].CopyFrom(
      tf.make_tensor_proto([length], shape=(1,)))

  return stub.Predict.future(request, timeout)

def main():
  parser = argparse.ArgumentParser(description="Translation client example")
  parser.add_argument("--model_name", required=True,
                      help="model name")
  parser.add_argument("--host", default="localhost",
                      help="model server host")
  parser.add_argument("--port", type=int, default=9000,
                      help="model server port")
  parser.add_argument("--timeout", type=float, default=10.0,
                      help="request timeout")
  args = parser.parse_args()

  channel = implementations.insecure_channel(args.host, args.port)
  stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

  batch_tokens = [
      ["Hello", "world", "!"],
      ["My", "name", "is", "John", "."],
      ["I", "live", "on", "the", "West", "coast", "."]]

  futures = []
  for tokens in batch_tokens:
    future = translate(stub, args.model_name, tokens, timeout=args.timeout)
    futures.append(future)

  for tokens, future in zip(batch_tokens, futures):
    result = parse_translation_result(future.result())
    print("{} ||| {}".format(" ".join(tokens), " ".join(result)))


if __name__ == "__main__":
  main()
