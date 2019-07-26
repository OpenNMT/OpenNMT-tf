"""Define inputters reading from TFRecord files."""

import tensorflow as tf
import numpy as np

from opennmt.inputters.inputter import Inputter


class SequenceRecordInputter(Inputter):
  """Inputter that reads ``tf.train.SequenceExample``."""

  def __init__(self, input_depth, dtype=tf.float32):
    """Initializes the parameters of the record inputter.

    Args:
      input_depth: The depth dimension of the input vectors.
      dtype: The values type.
    """
    super(SequenceRecordInputter, self).__init__(dtype=dtype)
    self.input_depth = input_depth

  def make_dataset(self, data_file, training=None):
    if not isinstance(data_file, dict):
      data_file = dict(path=data_file)
    return tf.data.TFRecordDataset(
        data_file["path"],
        compression_type=data_file.get("compression"))

  def input_signature(self):
    return {
        "tensor": tf.TensorSpec([None, None, self.input_depth], self.dtype),
        "length": tf.TensorSpec([None], tf.int32)
    }

  def make_features(self, element=None, features=None, training=None):
    if features is None:
      features = {}
    if "tensor" in features:
      return features
    element = tf.io.parse_single_sequence_example(element, sequence_features={
        "values": tf.io.FixedLenSequenceFeature([self.input_depth], dtype=tf.float32)})
    tensor = element[1]["values"]
    features["length"] = tf.shape(tensor)[0]
    features["tensor"] = tf.cast(tensor, self.dtype)
    return features

  def call(self, features, training=None):
    return features["tensor"]


def write_sequence_record(vector, writer):
  """Writes a sequence vector as a TFRecord.

  Args:
    vector: A 2D Numpy array of shape :math:`[T, D]`.
    writer: A ``tf.io.TFRecordWriter``.
  """
  feature_list = tf.train.FeatureList(feature=[
      tf.train.Feature(float_list=tf.train.FloatList(value=values))
      for values in vector.astype(np.float32)])
  feature_lists = tf.train.FeatureLists(feature_list={"values": feature_list})
  example = tf.train.SequenceExample(feature_lists=feature_lists)
  writer.write(example.SerializeToString())

def create_sequence_records(vectors, path, compression=None):
  """Creates a TFRecord file of sequence vectors.

  Args:
    vectors: An iterable of 2D Numpy array of shape :math:`[T, D]`.
    path: The output TFRecord file.
    compression: Optional compression type, can be "GZIP" or "ZLIB".
  """
  writer = tf.io.TFRecordWriter(path, options=compression)
  for vector in vectors:
    write_sequence_record(vector, writer)
  writer.close()
