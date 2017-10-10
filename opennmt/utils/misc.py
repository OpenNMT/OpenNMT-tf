"""Various utility functions to use throughout the project."""

import tensorflow as tf


def count_lines(filename):
  """Returns the number of lines of the file `filename`."""
  with open(filename) as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def count_parameters():
  """Returns the total number of trainable parameters."""
  total = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    count = 1
    for dim in shape:
      count *= dim.value
    total += count
  return total

def extract_prefixed_keys(dictionary, prefix):
  """Returns a dictionary with all keys from `dictionary` that are prefixed
  with `prefix`.
  """
  sub_dict = {}
  for key, value in dictionary.items():
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
    for _, value in tensors.items():
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in tensors.items()
      }


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

def add_dict_to_collection(dict_, collection_name):
  """Adds a dictionary to a graph collection.

  Args:
    dict_: A dictionary of string keys to tensor values
    collection_name: The name of the collection to add the dictionary to
  """
  key_collection = collection_name + "_keys"
  value_collection = collection_name + "_values"
  for key, value in dict_.items():
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
