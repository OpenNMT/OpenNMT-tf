"""Define inputters reading from TFRecord files."""

import tensorflow as tf

from opennmt.inputters.inputter import Inputter
from opennmt.utils import compat


class SequenceRecordInputter(Inputter):
  """Inputter that reads variable-length tensors.

  Each record contains the following fields:

   * ``shape``: the shape of the tensor as a ``int64`` list.
   * ``values``: the flattened tensor values as a :obj:`dtype` list.

  Tensors are expected to be of shape ``[time, depth]``.
  """

  def __init__(self, dtype=tf.float32):
    """Initializes the parameters of the record inputter.

    Args:
      dtype: The values type.
    """
    super(SequenceRecordInputter, self).__init__(dtype=dtype)

  def make_dataset(self, data_file, training=None):
    first_record = next(compat.tf_compat(v1="python_io.tf_record_iterator")(data_file))
    first_record = tf.train.Example.FromString(first_record)
    shape = first_record.features.feature["shape"].int64_list.value
    self.input_depth = shape[-1]
    return tf.data.TFRecordDataset(data_file)

  def get_dataset_size(self, data_file):
    return sum(1 for _ in compat.tf_compat(v1="python_io.tf_record_iterator")(data_file))

  def get_receiver_tensors(self):
    return {
        "tensor": tf.placeholder(self.dtype, shape=(None, None, self.input_depth)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features
    tf_parse_example = compat.tf_compat(v2="io.parse_single_example", v1="parse_single_example")
    tf_var_len_feature = compat.tf_compat(v2="io.VarLenFeature", v1="VarLenFeature")
    example = tf_parse_example(element, features={
        "shape": tf_var_len_feature(tf.int64),
        "values": tf_var_len_feature(tf.float32)
    })
    values = example["values"].values
    shape = tf.cast(example["shape"].values, tf.int32)
    tensor = tf.reshape(values, shape)
    tensor.set_shape([None, self.input_depth])
    features["length"] = tf.shape(tensor)[0]
    features["tensor"] = tf.cast(tensor, self.dtype)
    return features

  def make_inputs(self, features, training=None):
    return features["tensor"]


def write_sequence_record(vector, writer):
  """Writes a vector as a TFRecord.

  Args:
    vector: A 2D Numpy float array.
    writer: A ``tf.python_io.TFRecordWriter``.
  """
  shape = list(vector.shape)
  values = vector.flatten().tolist()

  example = tf.train.Example(features=tf.train.Features(feature={
      "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
      "values": tf.train.Feature(float_list=tf.train.FloatList(value=values))
  }))

  writer.write(example.SerializeToString())
