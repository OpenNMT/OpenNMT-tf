import tensorflow as tf

from opennmt.utils import data


class DataTest(tf.test.TestCase):

  def testIrregularBatches(self):
    batch_size = 12
    dataset = tf.data.Dataset.range(batch_size * 2 - 1)
    dataset = dataset.map(lambda x: {"x": x, "y": x + 1})
    dataset = dataset.batch(batch_size)
    dataset = dataset.apply(data.filter_irregular_batches(batch_size))

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      single_element = sess.run(next_element)
      self.assertEqual(batch_size, single_element["x"].size)
      with self.assertRaises(tf.errors.OutOfRangeError):
        sess.run(next_element)

  def testRandomShard(self):
    dataset_size = 42
    shard_size = 3

    dataset = tf.data.Dataset.range(dataset_size)
    dataset = dataset.apply(data.random_shard(shard_size, dataset_size))

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      gather = []
      while True:
        try:
          gather.append(sess.run(next_element))
        except tf.errors.OutOfRangeError:
          break
      # Check that all elements are fetched.
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

    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      if filtered:
        with self.assertRaises(tf.errors.OutOfRangeError):
          sess.run(next_element)
      else:
        sess.run(next_element)

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
    features = tf.random_normal([num_examples], mean=12, stddev=6, seed=42)
    labels_diff = tf.random_normal([num_examples], mean=0, stddev=3, seed=42)
    labels = features + labels_diff

    features = tf.maximum(tf.to_int32(1), tf.to_int32(features))
    labels = tf.maximum(tf.to_int32(1), tf.to_int32(labels))

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(features),
        tf.data.Dataset.from_tensor_slices(labels)))
    dataset = dataset.apply(data.batch_parallel_dataset(
        batch_size,
        features_length_fn=lambda x: x,
        labels_length_fn=lambda x: x,
        **kwargs))

    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    with self.test_session() as sess:
      sess.run(iterator.initializer)
      check_fn(sess, next_element)

  def testBatchTrainDatasetSimple(self):
    def _check_fn(sess, next_element):
      features, labels = sess.run(next_element)
      self.assertEqual(64, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 64)

  def testBatchTrainDatasetMultiplier(self):
    def _check_fn(sess, next_element):
      features, labels = sess.run(next_element)
      self.assertEqual(30, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 10, batch_multiplier=3)

  def testBatchTrainDatasetBucket(self):
    def _check_fn(sess, next_element):
      for _ in range(20):
        features, labels = sess.run(next_element)
        length = [max(f, l) for f, l in zip(features, labels)]
        self.assertGreater(3, max(length) - min(length))
        self.assertGreaterEqual(64, features.shape[0])
    self._testBatchTrainDataset(_check_fn, 64, bucket_width=3)

  def testBatchTrainDatasetTokens(self):
    def _check_fn(sess, next_element):
      for _ in range(20):
        features, labels = sess.run(next_element)
        batch_size = features.shape[0]
        max_length = max(list(features) + list(labels))
        self.assertGreaterEqual(256, batch_size * max_length)
    self._testBatchTrainDataset(_check_fn, 256, batch_type="tokens", bucket_width=1)


if __name__ == "__main__":
  tf.test.main()
