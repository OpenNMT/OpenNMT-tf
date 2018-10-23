"""Various utility functions to use throughout the project."""

from __future__ import print_function

import os
import sys
import inspect
import six

import numpy as np
import tensorflow as tf


def print_bytes(str_as_bytes, stream=None):
  """Prints a string viewed as bytes.

  Args:
    str_as_bytes: The bytes to print.
    stream: The stream to print to (``sys.stdout`` if not set).
  """
  if stream is None:
    stream = sys.stdout
  write_buffer = stream.buffer if hasattr(stream, "buffer") else stream
  write_buffer.write(str_as_bytes)
  write_buffer.write(b"\n")
  stream.flush()

def format_translation_output(sentence,
                              score=None,
                              token_level_scores=None,
                              attention=None,
                              alignment_type=None):
  """Formats a translation output with possibly scores, alignments, etc., e.g:

  1.123214 ||| Hello world ||| 0.30907777 0.030488174 ||| 0-0 1-1

  Args:
    sentence: The translation to output.
    score: If set, attach the score.
    token_level_scores: If set, attach the token level scores.
    attention: The attention vector.
    alignment_type: The type of alignments to format (can be: "hard").
  """
  if score is not None:
    sentence = "%f ||| %s" % (score, sentence)
  if token_level_scores is not None:
    scores_str = " ".join("%f" % s for s in token_level_scores)
    sentence = "%s ||| %s" % (sentence, scores_str)
  if attention is not None and alignment_type is not None:
    if alignment_type == "hard":
      source_indices = np.argmax(attention, axis=-1)
      target_indices = range(attention.shape[0])
      pairs = ("%d-%d" % (src, tgt) for src, tgt in zip(source_indices, target_indices))
      sentence = "%s ||| %s" % (sentence, " ".join(pairs))
    else:
      raise ValueError("Invalid alignment type %s" % alignment_type)
  return sentence

def item_or_tuple(x):
  """Returns :obj:`x` as a tuple or its single element."""
  x = tuple(x)
  if len(x) == 1:
    return x[0]
  else:
    return x

def classes_in_module(module, public_only=False):
  """Returns a generator over the classes defined in :obj:`module`."""
  return (symbol for symbol in dir(module)
          if (inspect.isclass(getattr(module, symbol))
              and (not public_only or not symbol.startswith("_"))))

def get_third_party_dir():
  """Returns a path to the third_party directory."""
  utils_dir = os.path.dirname(__file__)
  opennmt_dir = os.path.dirname(utils_dir)
  root_dir = os.path.dirname(opennmt_dir)
  third_party_dir = os.path.join(root_dir, "third_party")
  if not os.path.isdir(third_party_dir):
    raise RuntimeError("no third_party directory found in {}".format(root_dir))
  return third_party_dir

def count_lines(filename):
  """Returns the number of lines of the file :obj:`filename`."""
  with tf.gfile.Open(filename, mode="rb") as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def count_parameters():
  """Returns the total number of trainable parameters."""
  return sum(variable.get_shape().num_elements() for variable in tf.trainable_variables())

def extract_prefixed_keys(dictionary, prefix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are prefixed
  with :obj:`prefix`.
  """
  sub_dict = {}
  for key, value in six.iteritems(dictionary):
    if key.startswith(prefix):
      original_key = key[len(prefix):]
      sub_dict[original_key] = value
  return sub_dict

def extract_batches(tensors):
  """Returns a generator to iterate on each batch of a Numpy array or dict of
  Numpy arrays."""
  if not isinstance(tensors, dict):
    for tensor in tensors:
      yield tensor
  else:
    batch_size = None
    for value in six.itervalues(tensors):
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in six.iteritems(tensors)
      }

def merge_dict(dict1, dict2):
  """Merges :obj:`dict2` into :obj:`dict1`.

  Args:
    dict1: The base dictionary.
    dict2: The dictionary to merge.

  Returns:
    The merged dictionary :obj:`dict1`.
  """
  for key, value in six.iteritems(dict2):
    if isinstance(value, dict):
      dict1[key] = merge_dict(dict1.get(key, {}), value)
    else:
      dict1[key] = value
  return dict1

# The next 2 functions come with the following license and copyright:

# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

def add_dict_to_collection(collection_name, dict_):
  """Adds a dictionary to a graph collection.

  Args:
    collection_name: The name of the collection to add the dictionary to
    dict_: A dictionary of string keys to tensor values
  """
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in six.iteritems(dict_):
    tf.add_to_collection(key_collection, key)
    tf.add_to_collection(value_collection, value)

def get_dict_from_collection(collection_name):
  """Gets a dictionary from a graph collection.

  Args:
    collection_name: A collection name to read a dictionary from

  Returns:
    A dictionary with string keys and tensor values
  """
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  keys = tf.get_collection(key_collection)
  values = tf.get_collection(value_collection)
  return dict(zip(keys, values))
