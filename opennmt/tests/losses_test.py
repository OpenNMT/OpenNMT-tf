import tensorflow as tf

from parameterized import parameterized

from opennmt.utils import losses


class LossesTest(tf.test.TestCase):
    def testCrossEntropySequenceLoss(self):
        logits = tf.constant(
            [
                [[0.1, 0.2, 0.9], [-1.2, 2.1, 0], [0.6, 0.3, 0.4]],
                [[-2.2, -0.2, -1.2], [2.3, 0.2, -0.1], [0.0, 0.1, 0.7]],
            ]
        )
        labels = tf.constant([[2, 1, 0], [1, 0, 2]], dtype=tf.int32)

        loss, training_norm, stats_norm = losses.cross_entropy_sequence_loss(
            logits, labels, training=True
        )
        self.assertNear(loss, 3.06985, 1e-5)
        self.assertEqual(training_norm, 2)
        self.assertEqual(stats_norm, 6)

        _, training_norm, stats_norm = losses.cross_entropy_sequence_loss(
            logits, labels, average_in_time=True, training=True
        )
        self.assertEqual(training_norm, 6)
        self.assertEqual(stats_norm, 6)

    def testMaskedCrossEntropySequenceLoss(self):
        logits = tf.constant(
            [
                [[0.1, 0.2, 0.9], [-1.2, 2.1, 0], [0.6, 0.3, 0.4]],
                [[-2.2, -0.2, -1.2], [2.3, 0.2, -0.1], [0.0, 0.1, 0.7]],
            ]
        )
        labels = tf.constant([[2, 1, 0], [1, 0, 2]], dtype=tf.int32)
        lengths = tf.constant([2, 1], dtype=tf.int32)

        loss, train_norm, stats_norm = losses.cross_entropy_sequence_loss(
            logits, labels, sequence_length=lengths, training=True
        )
        self.assertNear(loss, 1.22118, 1e-5)
        self.assertEqual(train_norm, 2)
        self.assertEqual(stats_norm, 3)

    def testWeightedAndMaskedCrossEntropySequenceLoss(self):
        logits = tf.constant(
            [
                [[0.1, 0.2, 0.9], [-1.2, 2.1, 0], [0.6, 0.3, 0.4]],
                [[-2.2, -0.2, -1.2], [2.3, 0.2, -0.1], [0.0, 0.1, 0.7]],
            ]
        )
        labels = tf.constant([[2, 1, 0], [1, 0, 2]], dtype=tf.int32)
        lengths = tf.constant([3, 2], dtype=tf.int32)
        weights = tf.constant([0.6, 1.2])

        loss, train_norm, stats_norm = losses.cross_entropy_sequence_loss(
            logits,
            labels,
            sequence_length=lengths,
            sequence_weight=weights,
            training=True,
        )
        self.assertNear(loss, 1.77306, 1e-5)
        self.assertNear(train_norm, tf.reduce_sum(weights), 1e-5)
        self.assertNear(
            stats_norm,
            tf.reduce_sum(tf.cast(lengths, tf.float32) * weights),
            1e-5,
        )

    @parameterized.expand(
        [
            ["l1", 1e-4],
            ["L1", 1e-4],
            ["l1", 1],
            ["l2", 1e-4],
            ["l1_l2", (1e-4, 1e-4)],
        ]
    )
    def testRegularization(self, type, scale):
        layer = tf.keras.layers.Dense(256)
        layer.build([None, 128])
        regularization = losses.regularization_penalty(
            type, scale, layer.trainable_variables
        )
        self.assertEqual(0, len(regularization.shape))
        self.evaluate(regularization)

    def testRegulaizationInvalidType(self):
        with self.assertRaises(ValueError):
            losses.regularization_penalty("l3", 1e-4, [])

    def testRegulaizationMissingScaleValue(self):
        with self.assertRaises(ValueError):
            losses.regularization_penalty("l1_l2", 1e-4, [])

    @parameterized.expand(
        [
            ["ce", False],
            ["mse", False],
            ["mse", True],
        ]
    )
    def testGuidedAlignmentCostUnderDistributionStrategy(self, cost_type, with_length):
        strategy = tf.distribute.MirroredStrategy(devices=["/cpu:0"])
        attention_probs = tf.random.uniform([2, 5, 6])
        gold_alignment = tf.random.uniform([2, 5, 6])
        if with_length:
            sequence_length = tf.constant([4, 5], dtype=tf.int32)
        else:
            sequence_length = None
        with strategy.scope():
            losses.guided_alignment_cost(
                attention_probs,
                gold_alignment,
                sequence_length=sequence_length,
                cost_type=cost_type,
            )


if __name__ == "__main__":
    tf.test.main()
