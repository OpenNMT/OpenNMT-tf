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


if __name__ == "__main__":
  tf.test.main()
