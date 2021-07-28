import tensorflow as tf

from parameterized import parameterized

from opennmt.data import text


class TextTest(tf.test.TestCase):
    def _testTokensToChars(self, tokens, expected_chars):
        expected_chars = tf.nest.map_structure(tf.compat.as_bytes, expected_chars)
        chars = text.tokens_to_chars(tf.constant(tokens, dtype=tf.string))
        self.assertListEqual(chars.to_list(), expected_chars)

    def testTokensToCharsEmpty(self):
        self._testTokensToChars([], [])

    def testTokensToCharsSingle(self):
        self._testTokensToChars(["Hello"], [["H", "e", "l", "l", "o"]])

    def testTokensToCharsMixed(self):
        self._testTokensToChars(
            ["Just", "a", "测试"], [["J", "u", "s", "t"], ["a"], ["测", "试"]]
        )

    @parameterized.expand(
        [
            [["a￭", "b", "c￭", "d", "￭e"], [["a￭", "b"], ["c￭", "d", "￭e"]]],
            [
                ["a", "￭", "b", "c￭", "d", "￭", "e"],
                [["a", "￭", "b"], ["c￭", "d", "￭", "e"]],
            ],
        ]
    )
    def testToWordsWithJoiner(self, tokens, expected):
        expected = tf.nest.map_structure(tf.compat.as_bytes, expected)
        tokens = tf.constant(tokens)
        words = text.tokens_to_words(tokens)
        self.assertAllEqual(words.to_list(), expected)

    @parameterized.expand(
        [
            [["▁a", "b", "▁c", "d", "e"], [["▁a", "b"], ["▁c", "d", "e"]]],
            [
                ["▁", "a", "b", "▁", "c", "d", "e"],
                [["▁", "a", "b"], ["▁", "c", "d", "e"]],
            ],
            [["a▁", "b", "c▁", "d", "e"], [["a▁"], ["b", "c▁"], ["d", "e"]]],
            [
                ["a", "▁b▁", "c", "d", "▁", "e"],
                [["a"], ["▁b▁"], ["c", "d"], ["▁", "e"]],
            ],
        ]
    )
    def testToWordsWithSpacer(self, tokens, expected):
        expected = tf.nest.map_structure(tf.compat.as_bytes, expected)
        tokens = tf.constant(tokens)
        words = text.tokens_to_words(tokens, subword_token="▁", is_spacer=True)
        self.assertAllEqual(words.to_list(), expected)

    def _testPharaohAlignments(self, line, lengths, expected_matrix):
        matrix = text.alignment_matrix_from_pharaoh(
            tf.constant(line), lengths[0], lengths[1], dtype=tf.int32
        )
        self.assertListEqual(expected_matrix, self.evaluate(matrix).tolist())

    def testPharaohAlignments(self):
        self._testPharaohAlignments("", [0, 0], [])
        self._testPharaohAlignments("0-0", [1, 1], [[1]])
        self._testPharaohAlignments(
            "0-0 1-1 2-2 3-3",
            [4, 4],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        )
        self._testPharaohAlignments(
            "0-0 1-1 2-3 3-2",
            [4, 4],
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        )
        self._testPharaohAlignments("0-0 1-2 1-1", [2, 3], [[1, 0], [0, 1], [0, 1]])
        self._testPharaohAlignments(
            "0-0 1-2 1-1 2-4",
            [3, 5],
            [[1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]],
        )

    @parameterized.expand([[True], [False]])
    def testInvalidPharaohAlignments(self, run_as_function):
        func = text.alignment_matrix_from_pharaoh
        if run_as_function:
            func = tf.function(func)

        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "source"):
            func(tf.constant("0-0 1-1 2-3 3-2"), 2, 4)
        with self.assertRaisesRegex(tf.errors.InvalidArgumentError, "target"):
            func(tf.constant("0-0 1-2 1-1 2-4"), 3, 4)


if __name__ == "__main__":
    tf.test.main()
