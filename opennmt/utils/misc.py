"""Various utility functions to use throughout the project."""

import collections
import copy
import sys
import inspect
import heapq
import os

import numpy as np
import tensorflow as tf

from tensorflow.python.training.tracking import graph_view


def get_devices(count=1, fallback_to_cpu=True):
  """Gets devices.

  Args:
    count: The number of devices to get.
    fallback_to_cpu: If ``True``, return CPU devices if no GPU is available.

  Returns:
    A list of device names.

  Raises:
    ValueError: if :obj:`count` is greater than the number of visible devices.
  """
  devices = tf.config.list_logical_devices(device_type="GPU")
  if not devices and fallback_to_cpu:
    devices = tf.config.list_logical_devices(device_type="CPU")
  if len(devices) < count:
    raise ValueError("Requested %d devices but only %d are visible" % (count, len(devices)))
  return devices[0:count]

def get_variables_name_mapping(root, root_key=None):
  """Returns mapping between variables and their name in the object-based
  representation.

  Args:
    root: The root layer.
    root_key: Key that was used to save :obj:`root`, if any.

  Returns:
    A dict mapping variables ref to names and a dict mapping variables name to
    variables.
  """
  # TODO: find a way to implement this function using public APIs.
  named_variables, _, _ = graph_view.ObjectGraphView(root).serialize_object_graph()
  variables_to_names = {}
  names_to_variables = {}
  for saveable_object in named_variables:
    variable = saveable_object.op
    if not hasattr(variable, "experimental_ref"):  # Ignore non Tensor-like objects.
      continue
    name = saveable_object.name
    if root_key is not None:
      name = "%s/%s" % (root_key, name)
    variables_to_names[variable.experimental_ref()] = name
    names_to_variables[name] = variable
  return variables_to_names, names_to_variables

def get_variable_name(variable, root, model_key="model"):
  """Gets the variable name in the object-based representation."""
  variables_to_names, _ = get_variables_name_mapping(root, root_key=model_key)
  # In case of a MirroredVariable, look up the primary variable
  variable = getattr(variable, "primary", variable)
  return variables_to_names.get(variable.experimental_ref())

def print_as_bytes(text, stream=None):
  """Prints a string as bytes to non rely on :obj:`stream` default encoding.

  Args:
    text: The text to print.
    stream: The stream to print to (``sys.stdout`` if not set).
  """
  if stream is None:
    stream = sys.stdout
  write_buffer = stream.buffer if hasattr(stream, "buffer") else stream
  write_buffer.write(tf.compat.as_bytes(text))
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
    alignment_type: The type of alignments to format (can be: "hard", "soft").
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
    elif alignment_type == "soft":
      vectors = []
      for vector in attention:
        vectors.append(" ".join("%.6f" % value for value in vector))
      sentence = "%s ||| %s" % (sentence, " ; ".join(vectors))
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

def count_lines(filename):
  """Returns the number of lines of the file :obj:`filename`."""
  with tf.io.gfile.GFile(filename, mode="rb") as f:
    i = 0
    for i, _ in enumerate(f):
      pass
    return i + 1

def is_gzip_file(filename):
  """Returns ``True`` if :obj:`filename` is a GZIP file."""
  return filename.endswith(".gz")

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)

  # If unknown rank, return dynamic shape
  if x.shape.dims is None:
    return tf.shape(x)

  static = x.shape.as_list()
  shape = tf.shape(x)

  ret = []
  for i, _ in enumerate(static):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

def index_structure(structure, path):
  """Follows :obj:`path` in a nested structure of objects, lists, and dicts."""
  for key in path.split("/"):
    if isinstance(structure, list):
      try:
        index = int(key)
        structure = structure[index] if index < len(structure) else None
      except ValueError:
        raise ValueError("Expected a list index, got %s instead" % key)
    elif isinstance(structure, dict):
      structure = structure.get(key)
    else:
      structure = getattr(structure, key, None)
    if structure is None:
      raise ValueError("Invalid path in structure: %s" % path)
  return structure

def clone_layer(layer):
  """Clones a layer."""
  return copy.deepcopy(layer)

def gather_all_layers(layer):
  """Returns all nested layer starting from :obj:`layer`."""
  layers = set()
  if not isinstance(layer, tf.Module):
    return layers
  layers.add(layer)
  for value in layer.__dict__.values():
    if isinstance(value, tf.Module):
      layers.update(gather_all_layers(value))
    elif isinstance(value, list):
      for sub_layer in value:
        layers.update(gather_all_layers(sub_layer))
  return layers

def set_dropout(root_layer, dropout):
  """Overrides all dropout values in :obj:`root_layer` and its descendants.

  Args:
    dropout: The dropout value to set.
  """
  for layer in gather_all_layers(root_layer):
    for attr, value in layer.__dict__.items():
      if isinstance(value, tf.keras.layers.Dropout):
        value.rate = dropout
      elif "dropout" in attr:
        setattr(layer, attr, dropout)

def extract_batches(tensors):
  """Returns a generator to iterate on each batch of a Numpy array or dict of
  Numpy arrays."""
  if not isinstance(tensors, dict):
    for tensor in tensors:
      yield tensor
  else:
    batch_size = None
    for value in tensors.values():
      batch_size = batch_size or value.shape[0]
    for b in range(batch_size):
      yield {
          key: value[b] for key, value in tensors.items()
      }

def extract_prefixed_keys(dictionary, prefix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are prefixed
  with :obj:`prefix`.
  """
  sub_dict = {}
  for key, value in dictionary.items():
    if key.startswith(prefix):
      original_key = key[len(prefix):]
      sub_dict[original_key] = value
  return sub_dict

def extract_suffixed_keys(dictionary, suffix):
  """Returns a dictionary with all keys from :obj:`dictionary` that are suffixed
  with :obj:`suffix`.
  """
  sub_dict = {}
  for key, value in dictionary.items():
    if key.endswith(suffix):
      original_key = key[:-len(suffix)]
      sub_dict[original_key] = value
  return sub_dict

def merge_dict(dict1, dict2):
  """Merges :obj:`dict2` into :obj:`dict1`.

  Args:
    dict1: The base dictionary.
    dict2: The dictionary to merge.

  Returns:
    The merged dictionary :obj:`dict1`.
  """
  for key, value in dict2.items():
    if isinstance(value, dict):
      dict1[key] = merge_dict(dict1.get(key, {}), value)
    else:
      dict1[key] = value
  return dict1

def read_summaries(event_dir, event_file_pattern="events.out.tfevents.*"):
  """Reads summaries from TensorFlow event files.

  Args:
    event_dir: Directory containing event files.
    event_file_pattern: The pattern to look for event files.

  Returns:
    A list of tuple (step, dict of summaries), sorted by step.
  """
  if not tf.io.gfile.exists(event_dir):
    return []
  summaries = collections.defaultdict(dict)
  for event_file in tf.io.gfile.glob(os.path.join(event_dir, event_file_pattern)):
    for event in tf.compat.v1.train.summary_iterator(event_file):
      if not event.HasField("summary"):
        continue
      for value in event.summary.value:
        tensor_proto = value.tensor
        tensor = tf.io.parse_tensor(
            tensor_proto.SerializeToString(), tf.as_dtype(tensor_proto.dtype))
        summaries[event.step][value.tag] = tf.get_static_value(tensor)
  return list(sorted(summaries.items(), key=lambda x: x[0]))

class OrderRestorer(object):
  """Helper class to restore out-of-order elements in order."""

  def __init__(self, index_fn, callback_fn):
    """Initializes this object.

    Args:
      index_fn: A callable mapping an element to a unique index.
      callback_fn: A callable taking an element that will be called in order.
    """
    self._index_fn = index_fn
    self._callback_fn = callback_fn
    self._next_index = 0
    self._elements = {}
    self._heap = []

  def _try_notify(self):
    while self._heap and self._heap[0] == self._next_index:
      index = heapq.heappop(self._heap)
      value = self._elements.pop(index)
      self._callback_fn(value)
      self._next_index += 1

  def push(self, x):
    """Push event :obj:`x`."""
    index = self._index_fn(x)
    if index < self._next_index:
      raise ValueError("Event index %d was already notified" % index)
    self._elements[index] = x
    heapq.heappush(self._heap, index)
    self._try_notify()
