"""Text manipulation."""

import tensorflow as tf


def tokens_to_chars(tokens):
    """Splits tokens into unicode characters.

    Example:

      >>> opennmt.data.tokens_to_chars(["hello", "world"])
      <tf.RaggedTensor [[b'h', b'e', b'l', b'l', b'o'], [b'w', b'o', b'r', b'l', b'd']]>

    Args:
      tokens: A string ``tf.Tensor`` of shape :math:`[T]`.

    Returns:
      The characters as a 2D string ``tf.RaggedTensor``.
    """
    return tf.strings.unicode_split(tokens, "UTF-8")


def tokens_to_words(tokens, subword_token="￭", is_spacer=None):
    """Converts a sequence of tokens to a sequence of words.

    Example:

      >>> opennmt.data.tokens_to_words(["He@@", "llo", "W@@", "orld", "@@!"], subword_token="@@")
      <tf.RaggedTensor [[b'He@@', b'llo'], [b'W@@', b'orld', b'@@!']]>

    Args:
      tokens: A 1D string ``tf.Tensor``.
      subword_token: The special token used by the subword tokenizer.
      is_spacer: Whether :obj:`subword_token` is used as a spacer (as in
        SentencePiece) or a joiner (as in BPE). If ``None``, will infer
        directly from :obj:`subword_token`.

    Returns:
      The words as a 2D string ``tf.RaggedTensor``.
    """
    if is_spacer is None:
        is_spacer = subword_token == "▁"
    if is_spacer:
        # First token implicitly starts with a spacer.
        left_and_single = tf.logical_or(
            tf.strings.regex_full_match(tokens, "%s.*" % subword_token),
            tf.one_hot(0, tf.shape(tokens)[0], on_value=True, off_value=False),
        )
        right = tf.strings.regex_full_match(tokens, ".+%s" % subword_token)
        word_start = tf.logical_or(tf.roll(right, shift=1, axis=0), left_and_single)
    else:
        right = tf.strings.regex_full_match(tokens, ".*%s" % subword_token)
        left = tf.strings.regex_full_match(tokens, "%s.*" % subword_token)
        subword = tf.logical_or(tf.roll(right, shift=1, axis=0), left)
        word_start = tf.logical_not(subword)
    start_indices = tf.squeeze(tf.where(word_start), -1)
    return tf.RaggedTensor.from_row_starts(tokens, start_indices)


def alignment_matrix_from_pharaoh(
    alignment_line, source_length, target_length, dtype=tf.float32
):
    """Parse Pharaoh alignments into an alignment matrix.

    Example:

      >>> opennmt.data.alignment_matrix_from_pharaoh("0-0 1-2 1-3 2-1", 3, 4)
      <tf.Tensor: shape=(4, 3), dtype=float32, numpy=
      array([[1., 0., 0.],
             [0., 0., 1.],
             [0., 1., 0.],
             [0., 1., 0.]], dtype=float32)>

    Args:
      alignment_line: A string ``tf.Tensor`` in the Pharaoh format.
      source_length: The length of the source sentence, without special symbols.
      target_length: The length of the target sentence, without special symbols.
      dtype: The output matrix dtype. Defaults to ``tf.float32`` for convenience
        when computing the guided alignment loss.

    Returns:
      The alignment matrix as a 2-D ``tf.Tensor`` of type :obj:`dtype` and shape
      ``[target_length, source_length]``, where ``[i, j] = 1`` if the ``i`` th
      target word is aligned with the ``j`` th source word.
    """
    align_pairs_str = tf.strings.split([alignment_line]).values
    align_pairs_flat_str = tf.strings.split(align_pairs_str, sep="-").values
    align_pairs_flat = tf.strings.to_number(align_pairs_flat_str, out_type=tf.int64)
    sparse_indices = tf.reshape(align_pairs_flat, [-1, 2])
    sparse_values = tf.ones([tf.shape(sparse_indices)[0]], dtype=dtype)
    source_length = tf.cast(source_length, tf.int64)
    target_length = tf.cast(target_length, tf.int64)
    maximum_ids = tf.reduce_max(sparse_indices, axis=0)
    assert_source_length = _assert_in_range(
        maximum_ids[0], source_length, alignment_line, "source"
    )
    assert_target_length = _assert_in_range(
        maximum_ids[1], target_length, alignment_line, "target"
    )
    with tf.control_dependencies([assert_source_length, assert_target_length]):
        alignment_matrix_sparse = tf.sparse.SparseTensor(
            sparse_indices, sparse_values, [source_length, target_length]
        )
        alignment_matrix = tf.sparse.to_dense(
            alignment_matrix_sparse, validate_indices=False
        )
        return tf.transpose(alignment_matrix)


def _assert_in_range(maximum_id, length, line, name):
    return tf.debugging.assert_less(
        maximum_id,
        length,
        message=tf.strings.format(
            "Length mismatch for alignment line {}: actual %s length is {}, but "
            "got %s id {} which is out of range. Please check that the alignment "
            "file is correctly aligned to the training file." % (name, name),
            [line, length, maximum_id],
        ),
    )
