# -*- coding: utf-8 -*-

"""Text manipulation."""

import tensorflow as tf


def tokens_to_chars(tokens):
  """Splits tokens into unicode characters.

  Args:
    tokens: A string ``tf.Tensor`` of shape :math:`[T]`.

  Returns:
    The characters as a 2D string ``tf.RaggedTensor``.
  """
  return tf.strings.unicode_split(tokens, "UTF-8")

def tokens_to_words(tokens, subword_token="￭", is_spacer=None):
  """Converts a sequence of tokens to a sequence of words.

  For example, if a BPE tokenization produces this sequence:

      ["He@@", "llo", "W@@", "orld", "@@!"]

  this function will return the tensor:

      [["He@@", "llo", ""], ["W@@", "orld", "@@!"]]

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
        tf.one_hot(0, tf.shape(tokens)[0], on_value=True, off_value=False))
    right = tf.strings.regex_full_match(tokens, ".+%s" % subword_token)
    word_start = tf.logical_or(tf.roll(right, shift=1, axis=0), left_and_single)
  else:
    right = tf.strings.regex_full_match(tokens, ".*%s" % subword_token)
    left = tf.strings.regex_full_match(tokens, "%s.*" % subword_token)
    subword = tf.logical_or(tf.roll(right, shift=1, axis=0), left)
    word_start = tf.logical_not(subword)
  start_indices = tf.squeeze(tf.where(word_start), -1)
  return tf.RaggedTensor.from_row_starts(tokens, start_indices)

def alignment_matrix_from_pharaoh(alignment_line,
                                  source_length,
                                  target_length,
                                  dtype=tf.float32):
  """Parse Pharaoh alignments into an alignment matrix.

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
  alignment_matrix_sparse = tf.sparse.SparseTensor(
      sparse_indices, sparse_values, [source_length, target_length])
  alignment_matrix = tf.sparse.to_dense(alignment_matrix_sparse, validate_indices=False)
  return tf.transpose(alignment_matrix)
