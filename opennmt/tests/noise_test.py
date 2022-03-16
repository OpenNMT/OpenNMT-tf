import tensorflow as tf

from parameterized import parameterized

from opennmt.data import noise


class NoiseTest(tf.test.TestCase):
    @parameterized.expand(
        [
            [["a", "b", "c", "e", "f"]],
            [[["a", "b", ""], ["c", "e", "f"]]],
        ]
    )
    def testWordDropoutNone(self, words):
        x = tf.constant(words)
        y = noise.WordDropout(0)(x)
        x, y = self.evaluate([x, y])
        self.assertAllEqual(x, y)

    @parameterized.expand(
        [
            [[]],
            [["a", "b", "c", "e", "f"]],
            [[["a", "b", ""], ["c", "e", "f"]]],
        ]
    )
    def testWordDropoutAll(self, words):
        x = tf.constant(words, dtype=tf.string)
        y = noise.WordDropout(1)(x)
        y = self.evaluate(y)
        self.assertEqual(y.shape[0], 1 if words else 0)  # At least one is not dropped.

    @parameterized.expand([[1], [2], [4], [5]])
    def testWordOmission(self, count):
        words = [["a", "b", ""], ["c", "", ""], ["d", "e", "f"], ["g", "", ""]]
        x = tf.constant(words, dtype=tf.string)
        y = noise.WordOmission(count)(x)
        y = self.evaluate(y)
        expected_omit_count = min(count, len(words) - 1)
        self.assertEqual(y.shape[0], len(words) - expected_omit_count)

    @parameterized.expand(
        [
            [["a", "b", "c"], ["d", "d", "d"]],
            [[["a", "b", ""], ["c", "e", "f"]], [["d", "", ""], ["d", "", ""]]],
        ]
    )
    def testWordReplacement(self, words, expected):
        expected = tf.constant(expected)
        words = tf.constant(words)
        words = noise.WordReplacement(1, filler="d")(words)
        words, expected = self.evaluate([words, expected])
        self.assertAllEqual(words, expected)

    @parameterized.expand([[0], [1], [3]])
    def testWordPermutation(self, k):
        x = tf.constant("0 1 2 3 4 5 6 7 8 9 10 11 12 13".split())
        y = noise.WordPermutation(k)(x)
        x, y = self.evaluate([x, y])
        if k == 0:
            self.assertAllEqual(y, x)
        else:
            for i, v in enumerate(y.tolist()):
                self.assertLess(abs(int(v) - i), k)

    @parameterized.expand(
        [
            [True, [["a￭", "b", "c￭", "d", "￭e"], ["a", "b", "c", "", ""]], [5, 3]],
            [False, [["a￭", "b", "c￭", "d", "￭e"], ["a", "b", "c", "", ""]], [5, 3]],
            [False, ["a￭", "b", "c￭", "d", "￭e"], None],
            [
                False,
                [
                    [["a￭", "b", "c￭", "d", "￭e"], ["a￭", "b", "c￭", "d", "￭e"]],
                    [["a", "b", "c", "", ""], ["a", "b", "c", "", ""]],
                ],
                [[5, 5], [3, 3]],
            ],
        ]
    )
    def testWordNoising(self, as_function, tokens, lengths):
        tokens = tf.constant(tokens)
        if lengths is not None:
            lengths = tf.constant(lengths, dtype=tf.int32)
        noiser = noise.WordNoiser()
        noiser.add(noise.WordDropout(0.1))
        noiser.add(noise.WordReplacement(0.1))
        noiser.add(noise.WordPermutation(3))
        noiser_fn = tf.function(noiser) if as_function else noiser
        noisy_tokens, noisy_lengths = noiser_fn(
            tokens, sequence_length=lengths, keep_shape=True
        )
        tokens, noisy_tokens = self.evaluate([tokens, noisy_tokens])
        self.assertAllEqual(noisy_tokens.shape, tokens.shape)

    def testWordNoisingRagged(self):
        tokens = tf.RaggedTensor.from_tensor(
            [["a￭", "b", "c￭", "d", "e"], ["a￭", "b￭", "c", "", ""]], padding=""
        )
        noiser = noise.WordNoiser()
        noiser.add(noise.WordReplacement(1, filler="f"))
        tokens = noiser(tokens)
        self.assertAllEqual(tokens.to_list(), [[b"f", b"f", b"f"], [b"f"]])


if __name__ == "__main__":
    tf.test.main()
