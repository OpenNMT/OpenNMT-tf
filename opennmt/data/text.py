"""Text manipulation."""

import tensorflow as tf

from opennmt.constants import PADDING_TOKEN


def tokens_to_chars(tokens, padding_value=PADDING_TOKEN):
  """Splits tokens into unicode characters.

  Args:
    tokens: A string ``tf.Tensor`` of shape :math:`[T]`.
    padding_value: The value to use for padding.

  Returns:
    The characters as a string ``tf.Tensor`` of shape :math:`[T, W]` and the
    length of each token as an int64 ``tf.Tensor``  of shape :math:`[T]`.
  """
  ragged = tf.strings.unicode_split(tokens, "UTF-8")
  chars = ragged.to_tensor(default_value=padding_value)
  lengths = ragged.row_lengths()
  return chars, lengths

def tokens_to_words(tokens, subword_token="￭", is_spacer=False):
  """Converts a sequence of tokens to a sequence of words.

  For example, if a BPE tokenization produces this sequence:

      ["He@@", "llo", "W@@", "orld", "@@!"]

  this function will return the tensor:

      [["He@@", "llo", ""], ["W@@", "orld", "@@!"]]

  Args:
    tokens: A 1D string ``tf.Tensor``.
    subword_token: The special token used by the subword tokenizer.
    is_spacer: Whether :obj:`subword_token` is used as a spacer (as in
      SentencePiece) or a joiner (as in BPE).

  Returns:
    A 2D string ``tf.Tensor``.
  """
  if is_spacer:
    subword = tf.strings.regex_full_match(tokens, "[^%s].*" % subword_token)
  else:
    right = tf.strings.regex_full_match(tokens, ".*%s" % subword_token)
    left = tf.strings.regex_full_match(tokens, "%s.*" % subword_token)
    subword = tf.logical_or(tf.roll(right, shift=1, axis=0), left)
  start = tf.logical_not(subword)
  start_indices = tf.squeeze(tf.where(start), -1)
  words = tf.RaggedTensor.from_row_starts(tokens, start_indices)
  return words.to_tensor()
