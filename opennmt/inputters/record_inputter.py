"""Define inputters reading from TFRecord files."""

import tensorflow as tf
import numpy as np

from opennmt.data import dataset as dataset_util
from opennmt.inputters.inputter import Inputter


class SequenceRecordInputter(Inputter):
    """Inputter that reads ``tf.train.SequenceExample``.

    See Also:
      :func:`opennmt.inputters.create_sequence_records` to generate a compatible
      dataset.
    """

    def __init__(self, input_depth, **kwargs):
        """Initializes the parameters of the record inputter.

        Args:
          input_depth: The depth dimension of the input vectors.
          **kwargs: Additional layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.input_depth = input_depth

    def make_dataset(self, data_file, training=None):
        return dataset_util.make_datasets(tf.data.TFRecordDataset, data_file)

    def input_signature(self):
        return {
            "tensor": tf.TensorSpec([None, None, self.input_depth], self.dtype),
            "length": tf.TensorSpec([None], tf.int32),
        }

    def make_features(self, element=None, features=None, training=None):
        if features is None:
            features = {}
        if "tensor" in features:
            return features
        _, feature_lists = tf.io.parse_single_sequence_example(
            element,
            sequence_features={
                "values": tf.io.FixedLenSequenceFeature(
                    [self.input_depth], dtype=tf.float32
                )
            },
        )
        tensor = feature_lists["values"]
        features["length"] = tf.shape(tensor)[0]
        features["tensor"] = tf.cast(tensor, self.dtype)
        return features

    def call(self, features, training=None):
        return features["tensor"]


def write_sequence_record(vector, writer):
    """Writes a sequence vector as a TFRecord.

    Args:
      vector: A 2D Numpy float array of shape :math:`[T, D]`.
      writer: A ``tf.io.TFRecordWriter``.

    See Also:
      - :class:`opennmt.inputters.SequenceRecordInputter`
      - :func:`opennmt.inputters.create_sequence_records`
    """
    feature_list = tf.train.FeatureList(
        feature=[
            tf.train.Feature(float_list=tf.train.FloatList(value=values))
            for values in vector.astype(np.float32)
        ]
    )
    feature_lists = tf.train.FeatureLists(feature_list={"values": feature_list})
    example = tf.train.SequenceExample(feature_lists=feature_lists)
    writer.write(example.SerializeToString())


def create_sequence_records(vectors, path, compression=None):
    """Creates a TFRecord file of sequence vectors.

    Args:
      vectors: An iterable of 2D Numpy float arrays of shape :math:`[T, D]`.
      path: The output TFRecord file.
      compression: Optional compression type, can be "GZIP".

    Returns:
      Path to the TFRecord file. In most cases this is the same as :obj:`path` but
      if GZIP compression is enabled, the ".gz" extension is added if not already
      present.

    Raises:
      ValueError: if :obj:`compression` is invalid.

    See Also:
      - :class:`opennmt.inputters.SequenceRecordInputter`
      - :func:`opennmt.inputters.write_sequence_record`
    """
    if compression is not None:
        if compression not in ("GZIP",):
            raise ValueError("invalid compression type: %s" % compression)
        if compression == "GZIP" and not path.endswith(".gz"):
            path = "%s.gz" % path
    writer = tf.io.TFRecordWriter(path, options=compression)
    for vector in vectors:
        write_sequence_record(vector, writer)
    writer.close()
    return path
