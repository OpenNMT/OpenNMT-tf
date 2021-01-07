import os

from parameterized import parameterized

import tensorflow as tf

from opennmt.data import dataset as dataset_util
from opennmt.tests import test_util


class DatasetTest(tf.test.TestCase):
    @parameterized.expand(
        [
            [(4, 2), None, (4 / 6, 2 / 6)],
            [(4, 2), (0.2, 0.4), (0.5, 0.5)],
        ]
    )
    def testNormalizeWeights(self, sizes, weights, expected_weights):
        datasets = list(map(tf.data.Dataset.range, sizes))
        weights = dataset_util.normalize_weights(datasets, weights=weights)
        for weight, expected_weight in zip(weights, expected_weights):
            self.assertAllClose(weight, expected_weight)

    @parameterized.expand(
        [
            [(1, 1, 1), None, (1 / 3, 1 / 3, 1 / 3)],
            [(1, 2, 6), None, (1 / 9, 2 / 9, 6 / 9)],
            [(1, 1, 1), (0.2, 0.7, 0.1), (0.2, 0.7, 0.1)],
            [(1, 1, 1), (0.4, 1.4, 0.2), (0.2, 0.7, 0.1)],
        ]
    )
    def testDatasetWeighting(self, sizes, weights, target_distribution):
        datasets = [
            tf.data.Dataset.from_tensor_slices([i] * size)
            for i, size in enumerate(sizes)
        ]
        if weights is not None:
            datasets = (datasets, weights)
        dataset = dataset_util.training_pipeline(
            batch_size=20, shuffle_buffer_size=5000
        )(datasets)
        counts = [0] * len(sizes)
        # Check that after 2000 batches we are close to the target distribution.
        for x in dataset.take(2000):
            for i, _ in enumerate(counts):
                counts[i] += int(tf.math.count_nonzero(x == i))
        total_count = sum(counts)
        for count, freq in zip(counts, target_distribution):
            self.assertNear(count / total_count, freq, 0.05)

    def testDatasetSize(self):
        path = test_util.make_data_file(
            os.path.join(self.get_temp_dir(), "file.txt"), list(map(str, range(15)))
        )
        dataset = tf.data.TextLineDataset(path)
        size = dataset_util.get_dataset_size(dataset)
        self.assertEqual(self.evaluate(size), 15)

    def testDatasetSizeInfinite(self):
        dataset = tf.data.Dataset.range(5).repeat()
        self.assertIsNone(dataset_util.get_dataset_size(dataset))

    def testIrregularBatches(self):
        batch_size = 12
        dataset = tf.data.Dataset.range(batch_size * 2 - 1)
        dataset = dataset.map(lambda x: {"x": x, "y": x + 1})
        dataset = dataset.batch(batch_size)
        dataset = dataset.apply(dataset_util.filter_irregular_batches(batch_size))
        iterator = iter(dataset)
        single_element = next(iterator)
        self.assertEqual(batch_size, single_element["x"].shape[0])
        with self.assertRaises(StopIteration):
            next(iterator)

    @parameterized.expand([[11, 5, 15], [10, 5, 10], [5, 20, 20]])
    def testMakeCardinalityMultipleOf(self, dataset_size, divisor, expected_size):
        dataset = tf.data.Dataset.range(dataset_size)
        dataset = dataset.apply(dataset_util.make_cardinality_multiple_of(divisor))
        self.assertLen(list(iter(dataset)), expected_size)

    def testRandomShard(self):
        dataset_size = 42
        shard_size = 3

        dataset = tf.data.Dataset.range(dataset_size)
        dataset = dataset.apply(dataset_util.random_shard(shard_size, dataset_size))
        gather = list(iter(dataset))
        self.assertAllEqual(list(range(dataset_size)), sorted(gather))

    def _testFilterByLength(
        self,
        features_length,
        labels_length,
        maximum_features_length=None,
        maximum_labels_length=None,
        filtered=True,
    ):
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensors(tf.constant(features_length)),
                tf.data.Dataset.from_tensors(tf.constant(labels_length)),
            )
        )
        dataset = dataset.apply(
            dataset_util.filter_examples_by_length(
                maximum_features_length=maximum_features_length,
                maximum_labels_length=maximum_labels_length,
                features_length_fn=lambda _: features_length,
                labels_length_fn=lambda _: labels_length,
            )
        )

        iterator = iter(dataset)
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
            1, 1, maximum_features_length=1, maximum_labels_length=1, filtered=False
        )
        self._testFilterByLength(
            2, 1, maximum_features_length=1, maximum_labels_length=1, filtered=True
        )
        self._testFilterByLength(
            1, 2, maximum_features_length=1, maximum_labels_length=1, filtered=True
        )

    def testFilterExamplesByLengthMultiSource(self):
        self._testFilterByLength(
            [1, 1],
            1,
            maximum_features_length=1,
            maximum_labels_length=1,
            filtered=False,
        )
        self._testFilterByLength(
            [1, 2],
            1,
            maximum_features_length=1,
            maximum_labels_length=1,
            filtered=False,
        )
        self._testFilterByLength(
            [1, 0], 1, maximum_features_length=1, maximum_labels_length=1, filtered=True
        )
        self._testFilterByLength(
            [1, 2],
            1,
            maximum_features_length=[1, 1],
            maximum_labels_length=1,
            filtered=True,
        )

    def _testBatchTrainDataset(self, check_fn, batch_size, **kwargs):
        num_examples = 1000
        features = tf.random.normal([num_examples], mean=12, stddev=6, seed=42)
        labels_diff = tf.random.normal([num_examples], mean=0, stddev=3, seed=42)
        labels = features + labels_diff

        features = tf.maximum(tf.cast(1, tf.int32), tf.cast(features, tf.int32))
        labels = tf.maximum(tf.cast(1, tf.int32), tf.cast(labels, tf.int32))

        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(features),
                tf.data.Dataset.from_tensor_slices(labels),
            )
        )
        dataset = dataset.apply(
            dataset_util.batch_sequence_dataset(
                batch_size, length_fn=[lambda x: x, lambda x: x], **kwargs
            )
        )

        iterator = iter(dataset)
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
            length_bucket_width=10,
        )

    def testBatchTrainDatasetBucket(self):
        def _check_fn(iterator):
            for _ in range(20):
                features, labels = next(iterator)
                length = [max(f, l) for f, l in zip(features, labels)]
                self.assertGreater(3, max(length) - min(length))
                self.assertGreaterEqual(64, features.shape[0])

        self._testBatchTrainDataset(_check_fn, 64, length_bucket_width=3)

    def testBatchTrainDatasetTokens(self):
        def _check_fn(iterator):
            for _ in range(20):
                features, labels = next(iterator)
                batch_size = features.shape[0]
                max_length = max(list(features) + list(labels))
                self.assertGreaterEqual(256, batch_size * max_length)

        self._testBatchTrainDataset(
            _check_fn, 256, batch_type="tokens", length_bucket_width=1
        )

    def testReorderInferDataset(self):
        dataset = tf.data.Dataset.from_tensor_slices([8, 2, 5, 6, 7, 1, 3, 9])
        dataset = dataset.map(lambda x: {"length": x})
        dataset = dataset.apply(
            dataset_util.inference_pipeline(
                3, length_bucket_width=3, length_fn=lambda x: x["length"]
            )
        )
        elements = list(iter(dataset))

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
