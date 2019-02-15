import tensorflow as tf

from opennmt.utils import compat, data


class DataTest(tf.test.TestCase):

  def _iterDataset(self, dataset):
    if compat.is_tf2():
      for element in dataset:
        yield element
    else:
      iterator = dataset.make_initializable_iterator()
      next_element = iterator.get_next()
      with self.test_session() as sess:
        sess.run(iterator.initializer)
        while True:
          try:
            yield sess.run(next_element)
          except tf.errors.OutOfRangeError:
            return

  def testIrregularBatches(self):
    batch_size = 12
    dataset = tf.data.Dataset.range(batch_size * 2 - 1)
    dataset = dataset.map(lambda x: {"x": x, "y": x + 1})
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(data.filter_irregular_batches(batch_size))
    iterator = self._iterDataset(dataset)
    single_element = next(iterator)
    self.assertEqual(batch_size, single_element["x"].shape[0])
    with self.assertRaises(StopIteration):
      next(iterator)

  def testRandomShard(self):
    dataset_size = 42
    shard_size = 3

    dataset = tf.data.Dataset.range(dataset_size)
    dataset = dataset.apply(data.random_shard(shard_size, dataset_size))
    gather = list(self._iterDataset(dataset))
    self.assertAllEqual(list(range(dataset_size)), sorted(gather))

  def _testFilterByLength(self,
                          features_length,
                          labels_length,
                          maximum_features_length=None,
                          maximum_labels_length=None,
                          filtered=True):
    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensors(tf.constant(features_length)),
        tf.data.Dataset.from_tensors(tf.constant(labels_length))))
    dataset = dataset.apply(data.filter_examples_by_length(
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        features_length_fn=lambda _: features_length,
        labels_length_fn=lambda _: labels_length))

    iterator = self._iterDataset(dataset)
    if filtered:
      with self.assertRaises(StopIteration):
        next(iterator)
    else:
      next(iterator)

  def testFilterByLengthNoUpperBound(self):
    self._testFilterByLength(1, 1, filtered=False)
    self._testFilterByLength(0, 1, filtered=True)
    self._testFilterByLength(1, 0, filtered=True)

  def testFilterExamplesByLength(self):
    self._testFilterByLength(
        1, 1, maximum_features_length=1, maximum_labels_length=1, filtered=False)
    self._testFilterByLength(
        2, 1, maximum_features_length=1, maximum_labels_length=1, filtered=True)
    self._testFilterByLength(
        1, 2, maximum_features_length=1, maximum_labels_length=1, filtered=True)

  def testFilterExamplesByLengthMultiSource(self):
    self._testFilterByLength(
        [1, 1], 1, maximum_features_length=1, maximum_labels_length=1, filtered=False)
    self._testFilterByLength(
        [1, 2], 1, maximum_features_length=1, maximum_labels_length=1, filtered=False)
    self._testFilterByLength(
        [1, 0], 1, maximum_features_length=1, maximum_labels_length=1, filtered=True)
    self._testFilterByLength(
        [1, 2], 1, maximum_features_length=[1, 1], maximum_labels_length=1, filtered=True)

  def _testBatchTrainDataset(self, check_fn, batch_size, **kwargs):
    num_examples = 1000
    random_normal = compat.tf_compat(v2="random.normal", v1="random_normal")
    features = random_normal([num_examples], mean=12, stddev=6, seed=42)
    labels_diff = random_normal([num_examples], mean=0, stddev=3, seed=42)
    labels = features + labels_diff

    features = tf.maximum(tf.cast(1, tf.int32), tf.cast(features, tf.int32))
    labels = tf.maximum(tf.cast(1, tf.int32), tf.cast(labels, tf.int32))

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(features),
        tf.data.Dataset.from_tensor_slices(labels)))
    dataset = dataset.apply(data.batch_parallel_dataset(
        batch_size,
        features_length_fn=lambda x: x,
        labels_length_fn=lambda x: x,
        **kwargs))

    iterator = self._iterDataset(dataset)
    check_fn(iterator)

  def testBatchTrainDatasetSimple(self):
    def _check_fn(iterator):
      features, labels = next(iterator)
      self.assertEqual(64, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 64)

  def testBatchTrainDatasetMultiplier(self):
    def _check_fn(iterator):
      features, labels = next(iterator)
      self.assertEqual(30, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 10, batch_multiplier=3)

  def testBatchTrainDatasetMultiple(self):
    def _check_fn(iterator):
      features, labels = next(iterator)
      self.assertEqual(features.shape[0] % 3, 0)
    self._testBatchTrainDataset(
        _check_fn,
        1024,
        batch_type="tokens",
        batch_size_multiple=3,
        bucket_width=10)

  def testBatchTrainDatasetBucket(self):
    def _check_fn(iterator):
      for _ in range(20):
        features, labels = next(iterator)
        length = [max(f, l) for f, l in zip(features, labels)]
        self.assertGreater(3, max(length) - min(length))
        self.assertGreaterEqual(64, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 64, bucket_width=3)

  def testBatchTrainDatasetTokens(self):
    def _check_fn(iterator):
      for _ in range(20):
        features, labels = next(iterator)
        batch_size = features.shape[0]
        max_length = max(list(features) + list(labels))
        self.assertGreaterEqual(256, batch_size * max_length)
    self._testBatchTrainDataset(_check_fn, 256, batch_type="tokens", bucket_width=1)

  def testReorderInferDataset(self):
    dataset = tf.data.Dataset.from_tensor_slices([8, 2, 5, 6, 7, 1, 3, 9])
    dataset = dataset.map(lambda x: {"length": x})
    dataset = data.inference_pipeline(
        dataset, 3, bucket_width=3, length_fn=lambda x: x["length"])
    elements = list(self._iterDataset(dataset))

    def _check_element(element, length, index):
      self.assertAllEqual(element["length"], length)
      self.assertAllEqual(element["index"], index)

    self.assertEqual(len(elements), 4)
    _check_element(elements[0], [8, 6, 7], [0, 3, 4])
    _check_element(elements[1], [2, 1], [1, 5])
    _check_element(elements[2], [5, 3], [2, 6])
    _check_element(elements[3], [9], [7])


if __name__ == "__main__":
  tf.test.main()
