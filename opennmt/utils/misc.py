"""Various utility functions to use throughout the project."""

from __future__ import print_function

import os
import sys
import inspect
import six

import tensorflow as tf


def print_bytes(str_as_bytes, stream=None):
  """Prints a string viewed as bytes.

  This function calls ``decode()`` depending on the output stream encoding.

  Args:
    str_as_bytes: The bytes to print.
    stream: The stream to print to (``sys.stdout`` if not set).
  """
  encoding = None
  if stream is not None:
    encoding = stream.encoding
  if encoding is None:
    encoding = sys.getdefaultencoding()
  text = str_as_bytes.decode(encoding) if encoding != "ascii" else str_as_bytes
  print(text, file=stream)
  if stream is not None:
    stream.flush()

def item_or_tuple(x):
  """Returns :obj:`x` as a tuple or its single element."""
  x = tuple(x)
  if len(x) == 1:
    return x[0]
  else:
    return x

def classes_in_module(module):
  """Returns a generator over the classes defined in :obj:`module`."""
  return (symbol for symbol in dir(module) if inspect.isclass(getattr(module, symbol)))

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
  total = 0
  for variable in tf.trainable_variables():
    shape = variable.get_shape()
    count = 1
    for dim in shape:
      count *= dim.value
    total += count
  return total

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
