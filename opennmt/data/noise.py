"""Noise modules."""

import abc

import tensorflow as tf

from opennmt import constants
from opennmt.data import text
from opennmt.utils import misc


class WordNoiser(object):
    """Applies noise to words sequences."""

    def __init__(self, noises=None, subword_token="￭", is_spacer=None):
        """Initializes the noising class.

        Args:
          noises: A list of :class:`opennmt.data.Noise` instances to apply
            sequentially.
          subword_token: The special token used by the subword tokenizer. This is
            required when the noise should be applied at the word level and not the
            subword level.
          is_spacer: Whether :obj:`subword_token` is used as a spacer (as in
            SentencePiece) or a joiner (as in BPE). If ``None``, will infer
            directly from :obj:`subword_token`.

        See Also:
          :func:`opennmt.data.tokens_to_words`
        """
        if noises is None:
            noises = []
        self.noises = noises
        self.subword_token = subword_token
        self.is_spacer = is_spacer

    def add(self, noise):
        """Adds a noise to apply."""
        self.noises.append(noise)

    def __call__(self, tokens, sequence_length=None, keep_shape=False):
        """Applies noise on :obj:`tokens`.

        Args:
          tokens: A string ``tf.Tensor`` or batch of string ``tf.Tensor``.
          sequence_length: When :obj:`tokens` is a batch, the length of each
            sequence in the batch.
          keep_shape: Ensure that the shape is kept. Otherwise, fit the shape to the
            new lengths.

        Returns:
          A tuple with the noisy version of :obj:`tokens` and the new lengths.

        Raises:
          ValueError: if :obj:`tokens` is a batch of string but
            :obj:`sequence_length` is not passed.
        """
        with tf.device("cpu:0"):
            return self._call(tokens, sequence_length, keep_shape)

    def _call(self, tokens, sequence_length, keep_shape):
        rank = tokens.shape.ndims
        if rank == 1:
            input_length = tf.shape(tokens)[0]
            if sequence_length is not None:
                tokens = tokens[:sequence_length]
            else:
                tokens = tokens[: tf.math.count_nonzero(tokens)]
            words = text.tokens_to_words(
                tokens, subword_token=self.subword_token, is_spacer=self.is_spacer
            )
            words = words.to_tensor()
            for noise in self.noises:
                words = noise(words)
            outputs = tf.RaggedTensor.from_tensor(words, padding="").flat_values
            output_length = tf.shape(outputs)[0]
            if keep_shape:
                outputs = tf.pad(outputs, [[0, input_length - output_length]])
            return outputs, output_length
        elif rank == 2:
            if sequence_length is None:
                raise ValueError("sequence_length must be passed for 2D inputs")
            tokens, sequence_length = tf.map_fn(
                lambda arg: self._call(*arg, keep_shape=True), (tokens, sequence_length)
            )
            if not keep_shape:
                tokens = tokens[:, : tf.reduce_max(sequence_length)]
            return tokens, sequence_length
        else:
            if sequence_length is None:
                raise ValueError("sequence_length must be passed for ND inputs")
            original_shape = misc.shape_list(tokens)
            tokens = tf.reshape(tokens, [-1, original_shape[-1]])
            sequence_length = tf.reshape(sequence_length, [-1])
            tokens, sequence_length = self._call(
                tokens, sequence_length, keep_shape=keep_shape
            )
            tokens = tf.reshape(tokens, original_shape[:-1] + [-1])
            sequence_length = tf.reshape(sequence_length, original_shape[:-1])
            return tokens, sequence_length


class Noise(abc.ABC):
    """Base class for noise modules."""

    def __call__(self, words):
        """Applies noise on a sequence of words.

        Args:
          words: The sequence of words as a string ``tf.Tensor``. If it has 2
            dimensions, each row represents a word that possibly contains multiple
            tokens.

        Returns:
          A noisy version of :obj:`words`.

        Raises:
          ValueError: if :obj:`words` has a rank greater than 2.
        """
        if words.shape.ndims > 2:
            raise ValueError("Noise only supports tensors of rank 2 or less")
        inputs = words
        if words.shape.ndims == 1:
            inputs = tf.expand_dims(inputs, 1)
        num_words = tf.shape(inputs)[0]
        outputs = tf.cond(
            tf.math.equal(num_words, 0),
            true_fn=lambda: inputs,
            false_fn=lambda: self._apply(inputs),
        )
        if words.shape.ndims == 1:
            outputs = tf.squeeze(outputs, 1)
        return outputs

    @abc.abstractmethod
    def _apply(self, words):
        """Applies noise on a sequence of words.

        Args:
          words: A 2D string ``tf.Tensor`` where each row represents a word that
            possibly contains multiple tokens.

        Returns:
          A noisy version of :obj:`words`.
        """
        raise NotImplementedError()


class WordDropout(Noise):
    """Randomly drops words in a sequence.

    Example:

      >>> noise = opennmt.data.WordDropout(0.5)
      >>> words = tf.constant(["a", "b", "c"])
      >>> noise(words).numpy()
      array([b'a', b'b'], dtype=object)
    """

    def __init__(self, dropout):
        """Initializes the noise module.

        Args:
          dropout: The probability to drop word.
        """
        self.dropout = dropout

    def _apply(self, words):
        if self.dropout == 0:
            return tf.identity(words)
        num_words = tf.shape(words, out_type=tf.int64)[0]
        keep_mask = random_mask([num_words], 1 - self.dropout)
        keep_ind = tf.where(keep_mask)
        # Keep at least one word.
        keep_ind = tf.cond(
            tf.equal(tf.shape(keep_ind)[0], 0),
            true_fn=lambda: tf.random.uniform([1], maxval=num_words, dtype=tf.int64),
            false_fn=lambda: tf.squeeze(keep_ind, -1),
        )
        return tf.gather(words, keep_ind)


class WordOmission(Noise):
    """Randomly omits words in a sequence.

    This is different than :class:`opennmt.data.WordDropout` as it drops a
    fixed number of words.

    Example:

      >>> noise = opennmt.data.WordOmission(1)
      >>> words = tf.constant(["a", "b", "c"])
      >>> noise(words).numpy()
      array([b'b', b'c'], dtype=object)
    """

    def __init__(self, count):
        """Initializes the noise module.

        Args:
          count: The number of words to omit.
        """
        self.count = count

    def _apply(self, words):
        if self.count == 0:
            return tf.identity(words)
        num_words = tf.shape(words)[0]
        indices = tf.range(num_words)
        shuffle_indices = tf.random.shuffle(indices)
        keep_count = tf.maximum(num_words - self.count, 1)
        keep_indices = tf.sort(shuffle_indices[:keep_count])
        return tf.gather(words, keep_indices)


class WordReplacement(Noise):
    """Randomly replaces words.

    Example:

      >>> noise = opennmt.data.WordReplacement(0.5)
      >>> words = tf.constant(["a", "b", "c"])
      >>> noise(words).numpy()
      array([b'a', b'<unk>', b'c'], dtype=object)
    """

    def __init__(self, probability, filler=constants.UNKNOWN_TOKEN):
        """Initializes the noise module.

        Args:
          probability: The probability to replace words.
          filler: The replacement token.
        """
        self.probability = probability
        self.filler = filler

    def _apply(self, words):
        if self.probability == 0:
            return tf.identity(words)
        shape = tf.shape(words)
        replace_mask = random_mask(shape[:1], self.probability)
        filler = tf.fill([shape[0], 1], self.filler)
        filler = tf.pad(filler, [[0, 0], [0, shape[-1] - 1]])
        return tf.where(
            tf.broadcast_to(tf.expand_dims(replace_mask, -1), tf.shape(words)),
            x=filler,
            y=words,
        )


class WordPermutation(Noise):
    """Randomly permutes words in a sequence with a maximum distance.

    Example:

      >>> noise = opennmt.data.WordPermutation(3)
      >>> words = tf.constant(["0", "1", "2", "3", "4", "5", "6"])
      >>> noise(words).numpy()
      array([b'1', b'0', b'2', b'4', b'3', b'6', b'5'], dtype=object)
    """

    def __init__(self, max_distance):
        """Initializes the noise module.

        Args:
          max_distance: The maximum permutation distance.
        """
        self.max_distance = max_distance

    def _apply(self, words):
        if self.max_distance == 0:
            return tf.identity(words)
        num_words = tf.shape(words)[0]
        offset = tf.random.uniform([num_words], maxval=1) * (self.max_distance + 1)
        offset = tf.cast(offset, num_words.dtype)
        new_pos = tf.argsort(tf.range(num_words) + offset)
        return tf.gather(words, new_pos)


def random_mask(shape, probability):
    """Generates a random boolean mask.

    Args:
      shape: The mask shape.
      probability: The probability to select an element.

    Returns:
      A boolean mask with shape :obj:`shape`.
    """
    probs = tf.random.uniform(shape, maxval=1)
    mask = tf.math.less(probs, probability)
    return mask
