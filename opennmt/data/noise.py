"""Noise modules."""

import abc

import tensorflow as tf

from opennmt import constants
from opennmt.data import text
from opennmt.utils import misc


class WordNoiser:
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

    def __call__(
        self, tokens, sequence_length=None, keep_shape=False, probability=None
    ):
        """Applies noise on :obj:`tokens`.

        Args:
          tokens: A string ``tf.Tensor``, a batch of string ``tf.Tensor``, or a
            string ``tf.RaggedTensor``.
          sequence_length: When :obj:`tokens` is a dense tensor, the length of
            each sequence in the batch.
          keep_shape: Ensure that the original dense shape is kept. Otherwise,
            fit the shape to the new lengths.
          probability: Probability to apply noise on each example.

        Returns:
          If :obj:`tokens` is a ``tf.RaggedTensor``, the method returns the
          noisy tokens as a ``tf.RaggedTensor``, otherwise it returns a tuple
          with the noisy tokens as a ``tf.Tensor`` and the new lengths.

        Raises:
          ValueError: if :obj:`tokens` is a batch of string but
            :obj:`sequence_length` is not passed.
          ValueError: if :obj:`keep_shape` is ``True`` but :obj:`tokens` is a
            ``tf.RaggedTensor``.
        """
        if probability is None:
            probability = 1
        with tf.device("cpu:0"):
            return self._call(tokens, sequence_length, keep_shape, probability)

    def _call(self, tokens, sequence_length, keep_shape, probability):
        input_rank = tokens.shape.rank
        input_is_ragged = isinstance(tokens, tf.RaggedTensor)
        batch_shape = None

        if input_is_ragged:
            if keep_shape:
                raise ValueError("keep_shape is not compatible with a ragged input")
            ragged_tokens = tokens
        elif input_rank == 1:
            tokens = tf.expand_dims(tokens, 0)
            ragged_tokens = tf.RaggedTensor.from_tensor(tokens)
        elif input_rank >= 2:
            if sequence_length is None:
                raise ValueError("sequence_length must be passed for ND dense inputs")
            if input_rank > 2:
                input_shape = misc.shape_list(tokens)
                batch_shape = input_shape[:-1]
                tokens = tf.reshape(tokens, [-1, input_shape[-1]])
                sequence_length = tf.reshape(sequence_length, [-1])
            ragged_tokens = tf.RaggedTensor.from_tensor(tokens, lengths=sequence_length)

        noisy_tokens = tf.map_fn(
            lambda tokens: self._maybe_apply_noise(tokens, probability),
            ragged_tokens,
            fn_output_signature=tf.RaggedTensorSpec(
                shape=[None], dtype=ragged_tokens.dtype, ragged_rank=0
            ),
        )

        if input_is_ragged:
            return noisy_tokens

        new_lengths = tf.cast(noisy_tokens.row_lengths(), tf.int32)
        noisy_tokens = noisy_tokens.to_tensor(
            shape=tf.shape(tokens) if keep_shape else None
        )
        if input_rank == 1:
            new_lengths = new_lengths[0]
            noisy_tokens = noisy_tokens[0]
        elif batch_shape is not None:
            noisy_tokens = tf.reshape(noisy_tokens, batch_shape + [-1])
            new_lengths = tf.reshape(new_lengths, batch_shape)

        return noisy_tokens, new_lengths

    def _maybe_apply_noise(self, tokens, probability):
        if probability == 1:
            return self._apply_noise(tokens)
        elif probability == 0:
            return tokens
        else:
            return tf.cond(
                random_mask([], probability),
                true_fn=lambda: self._apply_noise(tokens),
                false_fn=lambda: tokens,
            )

    def _apply_noise(self, tokens):
        words = text.tokens_to_words(
            tokens, subword_token=self.subword_token, is_spacer=self.is_spacer
        )
        words = words.to_tensor()
        for noise in self.noises:
            words = noise(words)
        tokens = tf.RaggedTensor.from_tensor(words, padding="").flat_values
        return tokens


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
