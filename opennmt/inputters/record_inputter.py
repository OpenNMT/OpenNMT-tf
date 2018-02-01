"""Define inputters reading from TFRecord files."""

import tensorflow as tf

from opennmt.inputters.inputter import Inputter


class SequenceRecordInputter(Inputter):
  """Inputter that reads variable-length tensors.

  Each record contains the following fields:

   * ``shape``: the shape of the tensor as a ``int64`` list.
   * ``values``: the flattened tensor values as a :obj:`dtype` list.

  Tensors are expected to be of shape ``[time, depth]``.
  """

  def __init__(self, input_depth_key, dtype=tf.float32):
    """Initializes the parameters of the record inputter.

    Args:
      input_depth_key: The data configuration key of the input depth value
        which has to be known statically.
      dtype: The values type.
    """
    super(SequenceRecordInputter, self).__init__(dtype=dtype)
    self.input_depth_key = input_depth_key

  def initialize(self, metadata):
    self.input_depth = metadata[self.input_depth_key]

  def get_length(self, data):
    return data["length"]

  def make_dataset(self, data_file):
    return tf.data.TFRecordDataset(data_file)

  def _get_serving_input(self):
    receiver_tensors = {
        "tensor": tf.placeholder(self.dtype, shape=(None, None, self.input_depth)),
        "length": tf.placeholder(tf.int32, shape=(None,))
    }

    return receiver_tensors, receiver_tensors.copy()

  def _process(self, data):
    data = super(SequenceRecordInputter, self)._process(data)

    if "tensor" not in data:
      features = tf.parse_single_example(data["raw"], features={
          "shape": tf.VarLenFeature(tf.int64),
          "values": tf.VarLenFeature(self.dtype)
      })

      values = features["values"].values
      shape = tf.cast(features["shape"].values, tf.int32)

      tensor = tf.reshape(values, shape)
      length = tf.shape(tensor)[0]

      data = self.set_data_field(data, "tensor", tensor, padded_shape=[None, self.input_depth])
      data = self.set_data_field(data, "length", length, padded_shape=[])

    return data

  def _transform_data(self, data, mode):
    return self.transform(data["tensor"], mode)

  def transform(self, inputs, mode):
    return inputs
