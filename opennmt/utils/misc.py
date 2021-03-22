"""Various utility functions to use throughout the project."""

import collections
import copy
import sys
import functools
import heapq
import os
import io

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.training.tracking import graph_view

from opennmt.utils import compat


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
    device_type = "GPU"
    devices = tf.config.list_logical_devices(device_type=device_type)
    if not devices and fallback_to_cpu:
        tf.get_logger().warning("No GPU is detected, falling back to CPU")
        device_type = "CPU"
        devices = tf.config.list_logical_devices(device_type=device_type)
    if len(devices) < count:
        raise ValueError(
            "Requested %d %s devices but %d %s %s visible"
            % (
                count,
                device_type,
                len(devices),
                device_type,
                "is" if len(devices) == 1 else "are",
            )
        )
    return devices[0:count]


# TODO: clean mixed precision API when TensorFlow requirement is updated to >=2.4.
_set_global_policy = compat.tf_any(
    "keras.mixed_precision.set_global_policy",
    "keras.mixed_precision.experimental.set_policy",
)
_get_global_policy = compat.tf_any(
    "keras.mixed_precision.global_policy",
    "keras.mixed_precision.experimental.global_policy",
)


def enable_mixed_precision(force=False):
    """Globally enables mixed precision if the detected hardware supports it.

    Args:
      force: Set ``True`` to force mixed precision mode even if the hardware
        does not support it.

    Returns:
      A boolean to indicate whether mixed precision was enabled or not.
    """
    if not force:
        gpu_devices = tf.config.get_visible_devices("GPU")
        if not gpu_devices:
            tf.get_logger().warning("Mixed precision not enabled: no GPU is detected")
            return False

        gpu_details = tf.config.experimental.get_device_details(gpu_devices[0])
        compute_capability = gpu_details.get("compute_capability")
        if compute_capability is None:
            tf.get_logger().warning(
                "Mixed precision not enabled: a NVIDIA GPU is required"
            )
            return False
        if compute_capability < (7, 0):
            tf.get_logger().warning(
                "Mixed precision not enabled: a NVIDIA GPU with compute "
                "capability 7.0 or above is required, but the detected GPU "
                "has compute capability %d.%d" % compute_capability
            )
            return False

    _set_global_policy("mixed_float16")
    return True


def disable_mixed_precision():
    """Globally disables mixed precision."""
    _set_global_policy("float32")


def mixed_precision_enabled():
    """Returns ``True`` if mixed precision is enabled."""
    policy = _get_global_policy()
    return "float16" in policy.name


def get_variables_name_mapping(root, root_key=None):
    """Returns mapping between variables and their name in the object-based
    representation.

    Args:
      root: The root layer.
      root_key: Key that was used to save :obj:`root`, if any.

    Returns:
      A dict mapping names to variables.
    """
    # TODO: find a way to implement this function using public APIs.
    names_to_variables = {}
    _, path_to_root = graph_view.ObjectGraphView(root)._breadth_first_traversal()
    for path in path_to_root.values():
        if not path:
            continue
        variable = path[-1].ref
        if not isinstance(variable, tf.Variable):
            continue
        name = "%s/%s" % (
            "/".join(field.name for field in path),
            ".ATTRIBUTES/VARIABLE_VALUE",
        )
        if root_key is not None:
            name = "%s/%s" % (root_key, name)
        names_to_variables[name] = variable
    return names_to_variables


def get_variable_name(variable, root, model_key="model"):
    """Gets the variable name in the object-based representation."""
    names_to_variables = get_variables_name_mapping(root, root_key=model_key)
    for name, var in names_to_variables.items():
        if var is variable:
            return name
    return None


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


def format_translation_output(
    sentence, score=None, token_level_scores=None, attention=None, alignment_type=None
):
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
            pairs = (
                "%d-%d" % (src, tgt) for src, tgt in zip(source_indices, target_indices)
            )
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


def count_lines(filename, buffer_size=65536):
    """Returns the number of lines of the file :obj:`filename`."""
    with tf.io.gfile.GFile(filename, mode="rb") as f:
        num_lines = 0
        while True:
            data = f.read(buffer_size)
            if not data:
                return num_lines
            num_lines += data.count(b"\n")


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


def index_structure(structure, path, path_separator="/"):
    """Follows :obj:`path` in a nested structure of objects, lists, and dicts."""
    keys = path.split(path_separator)
    for i, key in enumerate(keys):
        current_path = "%s%s" % (path_separator, path_separator.join(keys[:i]))
        if isinstance(structure, list):
            try:
                index = int(key)
            except ValueError as e:
                raise ValueError(
                    "Object referenced by path '%s' is a list, but got non "
                    "integer index '%s'" % (current_path, key)
                ) from e
            if index < 0 or index >= len(structure):
                raise ValueError(
                    "List referenced by path '%s' has length %d, but got "
                    "out of range index %d" % (current_path, len(structure), index)
                )
            structure = structure[index]
        elif isinstance(structure, dict):
            structure = structure.get(key)
            if structure is None:
                raise ValueError(
                    "Dictionary referenced by path '%s' does not have the "
                    "key '%s'" % (current_path, key)
                )
        else:
            structure = getattr(structure, key, None)
            if structure is None:
                raise ValueError(
                    "Object referenced by path '%s' does not have the "
                    "attribute '%s'" % (current_path, key)
                )
    return structure


def clone_layer(layer):
    """Clones a layer."""
    return copy.deepcopy(layer)


def set_dropout(root_layer, dropout):
    """Overrides all dropout values in :obj:`root_layer` and its descendants.

    Args:
      dropout: The dropout value to set.

    Raises:
      ValueError: if :obj:`root_layer` is not a ``tf.Module``.
    """
    if not isinstance(root_layer, tf.Module):
        raise ValueError("Layer should be a tf.Module")
    for layer in (root_layer,) + root_layer.submodules:
        for attr, value in layer.__dict__.copy().items():
            if isinstance(value, tf.keras.layers.Dropout):
                value.rate = dropout
            elif "dropout" in attr:
                setattr(layer, attr, dropout)


def describe_layer(layer, name=None):
    """Returns a PyTorch-style description of the layer, for information or debug."""
    with io.StringIO() as output:
        _describe_layer(output, layer, name=name)
        return output.getvalue()


def _describe_layer(output, layer, name=None, indent=""):
    if indent:
        output.write(indent)
    if name:
        output.write("(%s): " % name)
    output.write("%s(" % layer.__class__.__name__)
    if isinstance(layer, list):
        children = list(enumerate(layer))
    else:
        children = _get_direct_children(layer)
    if not children:
        units = getattr(layer, "units", None)
        if units is not None:
            output.write("%d" % units)
    else:
        output.write("\n")
        for attr_name, child in children:
            _describe_layer(output, child, name=str(attr_name), indent=indent + "  ")
        if indent:
            output.write(indent)
    output.write(")\n")


def _get_direct_children(layer):
    children = []
    for name, attr in layer.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(attr, tf.Module) or (
            isinstance(attr, list) and attr and isinstance(attr[0], tf.Module)
        ):
            children.append((name, attr))
    return children


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
            yield {key: value[b] for key, value in tensors.items()}


def extract_prefixed_keys(dictionary, prefix):
    """Returns a dictionary with all keys from :obj:`dictionary` that are prefixed
    with :obj:`prefix`.
    """
    sub_dict = {}
    for key, value in dictionary.items():
        if key.startswith(prefix):
            original_key = key[len(prefix) :]
            sub_dict[original_key] = value
    return sub_dict


def extract_suffixed_keys(dictionary, suffix):
    """Returns a dictionary with all keys from :obj:`dictionary` that are suffixed
    with :obj:`suffix`.
    """
    sub_dict = {}
    for key, value in dictionary.items():
        if key.endswith(suffix):
            original_key = key[: -len(suffix)]
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
                    tensor_proto.SerializeToString(), tf.as_dtype(tensor_proto.dtype)
                )
                summaries[event.step][value.tag] = tf.get_static_value(tensor)
    return list(sorted(summaries.items(), key=lambda x: x[0]))


def disable_tfa_custom_ops(func):
    """A decorator that disables TensorFlow Addons custom ops in a function."""

    def _wrapper(*args, **kwargs):
        previous_value = tfa.options.TF_ADDONS_PY_OPS
        tfa.options.TF_ADDONS_PY_OPS = True
        try:
            outputs = func(*args, **kwargs)
        finally:
            tfa.options.TF_ADDONS_PY_OPS = previous_value
        return outputs

    return _wrapper


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

    @property
    def buffer_size(self):
        """Number of elements waiting to be notified."""
        return len(self._heap)

    @property
    def next_index(self):
        """The next index to be notified."""
        return self._next_index

    def _try_notify(self):
        old_index = self._next_index
        while self._heap and self._heap[0] == self._next_index:
            index = heapq.heappop(self._heap)
            value = self._elements.pop(index)
            self._callback_fn(value)
            self._next_index += 1
        return self._next_index != old_index

    def push(self, x):
        """Push event :obj:`x`."""
        index = self._index_fn(x)
        if index is None:
            self._callback_fn(x)
            return True
        if index < self._next_index:
            raise ValueError("Event index %d was already notified" % index)
        self._elements[index] = x
        heapq.heappush(self._heap, index)
        return self._try_notify()


class ClassRegistry(object):
    """Helper class to create a registry of classes."""

    def __init__(self, base_class=None):
        """Initializes the class registry.

        Args:
          base_class: Ensure that classes added to this registry are a subclass of
            :obj:`base_class`.
        """
        self._base_class = base_class
        self._registry = {}

    @property
    def class_names(self):
        """Class names registered in this registry."""
        return set(self._registry.keys())

    def register(self, cls=None, name=None, alias=None):
        """Registers a class.

        Args:
          cls: The class to register. If not set, this method returns a decorator for
            registration.
          name: The class name. Defaults to ``cls.__name__``.
          alias: An optional alias or list of alias for this class.

        Returns:
          :obj:`cls` if set, else a class decorator.

        Raises:
          TypeError: if :obj:`cls` does not extend the expected base class.
          ValueError: if the class name is already registered.
        """
        if cls is None:
            return functools.partial(self.register, name=name, alias=alias)
        if self._base_class is not None and not issubclass(cls, self._base_class):
            raise TypeError(
                "Class %s does not extend %s"
                % (cls.__name__, self._base_class.__name__)
            )
        if name is None:
            name = cls.__name__
        self._register(cls, name)
        if alias is not None:
            if not isinstance(alias, (list, tuple)):
                alias = (alias,)
            for alias_name in alias:
                self._register(cls, alias_name)
        return cls

    def _register(self, cls, name):
        if name in self._registry:
            raise ValueError("Class name %s is already registered" % name)
        self._registry[name] = cls

    def get(self, name):
        """Returns the class with name :obj:`name` or ``None`` if it does not exist
        in the registry.
        """
        return self._registry.get(name)


class RelativeConfig(collections.abc.Mapping):
    """Helper class to lookup keys relative to a prefix."""

    def __init__(self, config, prefix=None, config_name=None):
        """Initializes the relative configuration.

        Args:
          config: The configuration.
          prefix: The prefix. Keys will be looked up relative to this prefix.
          config_name: The name of the configuration, mostly used to make error
            messages more explicit.
        """
        self._config = config
        self._prefix = prefix or ""
        self._config_name = config_name

    def __getitem__(self, relative_key):
        absolute_key = "%s%s" % (self._prefix, relative_key)
        value = self._config.get(absolute_key)
        if value is not None:
            return value
        value = self._config.get(relative_key)
        if value is not None:
            return value
        raise KeyError(
            "Missing field '%s' in the %sconfiguration"
            % (absolute_key, self._config_name + " " if self._config_name else "")
        )

    def __len__(self):
        return len(self._config)

    def __iter__(self):
        return iter(self._config)
