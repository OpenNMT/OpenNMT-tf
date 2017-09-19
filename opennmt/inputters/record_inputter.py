"""Define inputters reading from TFRecord files."""

import tensorflow as tf

from opennmt.inputters.inputter import Inputter


class SequenceRecordInputter(Inputter):
  """Inputter that reads variable-length tensors.

  Each record contains the following fields:

   * `shape`: the shape of the tensor as a `int64` list.
   * `values`: the flattened tensor values as a `float32` list.

  Tensors are expected to be of shape `[time, depth]`.
  """

  def __init__(self, input_depth_key):
    """Initializes the parameters of the record inputter.

    Args:
      input_depth_key: The run configuration key of the input depth value
        which has to be known statically.
    """
    super(SequenceRecordInputter, self).__init__()
    self.input_depth_key = input_depth_key

  def initialize(self, metadata):
    self.input_depth = metadata[self.input_depth_key]

  def make_dataset(self, data_file):
    return tf.contrib.data.TFRecordDataset(data_file)

  def get_serving_input_receiver(self):
    placeholder = tf.placeholder(tf.float32, shape=(None, self.input_depth))
    features = {
      "tensor": placeholder,
      "length": tf.shape(placeholder)[0]
    }

    # TODO: support batch input during preprocessing.
    for key, value in features.items():
      features[key] = tf.expand_dims(value, 0)

    receiver_tensors = {"sequences": placeholder}
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

  def _process(self, data):
    data = super(SequenceRecordInputter, self)._process(data)

    if not "tensor" in data:
      features = tf.parse_single_example(data["raw"], features={
        "shape": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32)
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

  def _transform(self, inputs, mode, reuse=None):
    return inputs
