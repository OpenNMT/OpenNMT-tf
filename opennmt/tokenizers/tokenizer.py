"""Define base tokenizers."""

import sys
import abc
import json
import yaml

import tensorflow as tf

from opennmt.utils import misc


class Tokenizer(abc.ABC):
    """Base class for tokenizers."""

    @property
    def in_graph(self):
        """Returns ``True`` if this tokenizer can be run in graph (i.e. uses TensorFlow ops)."""
        return False

    def export_assets(self, asset_dir, asset_prefix=""):
        """Exports assets for this tokenizer.

        Args:
          asset_dir: The directory where assets can be written.
          asset_prefix: The prefix to attach to assets filename.

        Returns:
          A dictionary containing additional assets used by the tokenizer.
        """
        return {}

    def tokenize_stream(
        self, input_stream=sys.stdin, output_stream=sys.stdout, delimiter=" "
    ):
        """Tokenizes a stream of sentences.

        Args:
          input_stream: The input stream.
          output_stream: The output stream.
          delimiter: The token delimiter to use for text serialization.
        """
        for line in input_stream:
            line = line.strip()
            tokens = self.tokenize(line)
            merged_tokens = delimiter.join(tokens)
            misc.print_as_bytes(merged_tokens, stream=output_stream)

    def detokenize_stream(
        self, input_stream=sys.stdin, output_stream=sys.stdout, delimiter=" "
    ):
        """Detokenizes a stream of sentences.

        Args:
          input_stream: The input stream.
          output_stream: The output stream.
          delimiter: The token delimiter used for text serialization.
        """
        for line in input_stream:
            tokens = line.strip().split(delimiter)
            string = self.detokenize(tokens)
            misc.print_as_bytes(string, stream=output_stream)

    def tokenize(self, text):
        """Tokenizes text.

        Args:
          text: A string or batch of strings to tokenize as a ``tf.Tensor`` or
            Python values.

        Returns:
          - If :obj:`text` is a Python string, a list of Python strings.
          - If :obj:`text` is a list of Python strings, a list of list of Python
            strings.
          - If :obj:`text` is a 0-D ``tf.Tensor``, a 1-D ``tf.Tensor``.
          - If :obj:`text` is a 1-D ``tf.Tensor``, a 2-D ``tf.RaggedTensor``.

        Raises:
          ValueError: if the rank of :obj:`text` is greater than 1.
        """
        with tf.device("cpu:0"):
            return self._tokenize(text)

    def _tokenize(self, text):
        if tf.is_tensor(text):
            rank = len(text.shape)
            if rank == 0:
                return self._tokenize_tensor(text)
            elif rank == 1:
                return self._tokenize_batch_tensor(text)
            else:
                raise ValueError("Unsupported tensor rank %d for tokenization" % rank)
        elif isinstance(text, list):
            return list(map(self.tokenize, text))
        else:
            text = tf.compat.as_text(text)
            return self._tokenize_string(text)

    def detokenize(self, tokens, sequence_length=None):
        """Detokenizes tokens.

        The Tensor version supports batches of tokens.

        Args:
          tokens: Tokens or batch of tokens as a ``tf.Tensor``, ``tf.RaggedTensor``,
            or Python values.
          sequence_length: The length of each sequence. Required if :obj:`tokens`
            is a dense 2-D ``tf.Tensor``.

        Returns:
          - If :obj:`tokens` is a list of list of Python strings, a list of Python strings.
          - If :obj:`tokens` is a list of Python strings, a Python string.
          - If :obj:`tokens` is a N-D ``tf.Tensor`` (or ``tf.RaggedTensor``), a
            (N-1)-D ``tf.Tensor``.

        Raises:
          ValueError: if the rank of :obj:`tokens` is greater than 2.
          ValueError: if :obj:`tokens` is a 2-D dense ``tf.Tensor`` and
            :obj:`sequence_length` is not set.
        """
        with tf.device("cpu:0"):
            return self._detokenize(tokens, sequence_length)

    def _detokenize(self, tokens, sequence_length):
        if isinstance(tokens, tf.RaggedTensor):
            rank = len(tokens.shape)
            if rank == 1:
                return self._detokenize_tensor(tokens.values)
            elif rank == 2:
                return self._detokenize_ragged_tensor(tokens)
            else:
                raise ValueError(
                    "Unsupported RaggedTensor rank %d for detokenization" % rank
                )
        elif tf.is_tensor(tokens):
            rank = len(tokens.shape)
            if rank == 1:
                return self._detokenize_tensor(tokens)
            elif rank == 2:
                if sequence_length is None:
                    raise ValueError(
                        "sequence_length is required for Tensor detokenization"
                    )
                return self._detokenize_batch_tensor(tokens, sequence_length)
            else:
                raise ValueError("Unsupported tensor rank %d for detokenization" % rank)
        elif isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            return list(map(self.detokenize, tokens))
        else:
            tokens = [tf.compat.as_text(token) for token in tokens]
            return self._detokenize_string(tokens)

    def _tokenize_tensor(self, text):
        """Tokenizes a tensor.

        When not overriden, this default implementation calls the string-based
        tokenization.

        Args:
          text: A 1-D string ``tf.Tensor``.

        Returns:
          A 1-D string ``tf.Tensor``.
        """

        def _python_wrapper(string_t):
            string = tf.compat.as_text(string_t.numpy())
            tokens = self._tokenize_string(string)
            return tf.constant(tokens, dtype=tf.string)

        tokens = tf.py_function(_python_wrapper, [text], tf.string)
        tokens.set_shape([None])
        return tokens

    def _tokenize_batch_tensor(self, text):
        """Tokenizes a batch of texts.

        When not overriden, this default implementation calls _tokenize_tensor on
        each tensor within the batch.

        Args:
          text: A 1-D string ``tf.Tensor``.

        Returns:
          A 2-D string ``tf.RaggedTensor``.
        """
        # map_fn expects each output element to have the same shape, so join tokens with
        # spaces first and then split on spaces with a function returning a RaggedTensor.
        tokens = tf.map_fn(
            lambda x: tf.strings.reduce_join(
                self._tokenize_tensor(x), axis=0, separator=" "
            ),
            text,
        )
        return tf.strings.split(tokens, sep=" ")

    def _detokenize_tensor(self, tokens):
        """Detokenizes tokens.

        When not overriden, this default implementation calls the string-based
        detokenization.

        Args:
          tokens: A 1-D ``tf.Tensor``.

        Returns:
          A 0-D string ``tf.Tensor``.
        """

        def _python_wrapper(tokens_t):
            tokens = [tf.compat.as_text(s) for s in tokens_t.numpy()]
            string = self._detokenize_string(tokens)
            return tf.constant(string)

        text = tf.py_function(_python_wrapper, [tokens], tf.string)
        text.set_shape([])
        return text

    def _detokenize_batch_tensor(self, tokens, sequence_length):
        """Detokenizes a batch of tokens.

        When not overriden, this default implementation calls _detokenize_tensor on
        each tensor within the batch.

        Args:
          tokens: A 2-D ``tf.Tensor``.

        Returns:
          A 1-D string ``tf.Tensor``.
        """
        return tf.map_fn(
            lambda x: self._detokenize_tensor(x[0][: x[1]]),
            (tokens, sequence_length),
            dtype=tf.string,
        )

    def _detokenize_ragged_tensor(self, tokens):
        """Detokenizes a batch of tokens as a ``tf.RaggedTensor``

        When not overriden, this default implementation calls _detokenize_batch_tensor
        on the dense representation.

        Args:
          tokens: A 2-D ``tf.RaggedTensor``.

        Returns:
          A 1-D string ``tf.Tensor``.
        """
        return self._detokenize_batch_tensor(tokens.to_tensor(), tokens.row_lengths())

    @abc.abstractmethod
    def _tokenize_string(self, text):
        """Tokenizes a Python unicode string.

        This method should be thread-safe.

        Args:
          text: A Python unicode string.

        Returns:
          A list of Python unicode strings.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def _detokenize_string(self, tokens):
        """Detokenizes tokens.

        Args:
          tokens: A list of Python unicode strings.

        Returns:
          A unicode Python string.
        """
        raise NotImplementedError()


_TOKENIZERS_REGISTRY = misc.ClassRegistry(base_class=Tokenizer)

register_tokenizer = _TOKENIZERS_REGISTRY.register


def make_tokenizer(config=None):
    """Creates a tokenizer instance from the configuration.

    Args:
      config: Tokenization configuration as a Python dictionary, or a path to a
        YAML configuration file, or a JSON string.

    Returns:
      A :class:`opennmt.tokenizers.Tokenizer` instance.

    Raises:
      ValueError: if :obj:`config` is invalid.
    """
    if config:
        if isinstance(config, str):
            if tf.io.gfile.exists(config):
                with tf.io.gfile.GFile(config) as config_file:
                    config = yaml.safe_load(config_file)
            else:
                try:
                    config = json.loads(config)
                except json.JSONDecodeError:
                    pass
        if isinstance(config, dict):
            tokenizer_type = config.get("type")
            if tokenizer_type is None:
                tokenizer_type = "OpenNMTTokenizer"
                tokenizer_params = config
            else:
                tokenizer_params = config.get("params", {})
            tokenizer_class = _TOKENIZERS_REGISTRY.get(tokenizer_type)
            if tokenizer_class is None:
                raise ValueError(
                    "%s is not in list of accepted tokenizers: %s"
                    % (
                        tokenizer_type,
                        ", ".join(sorted(_TOKENIZERS_REGISTRY.class_names)),
                    )
                )
            tokenizer = tokenizer_class(**tokenizer_params)
        else:
            raise ValueError("Invalid tokenization configuration: %s" % str(config))
    else:
        # If the tokenization was not configured, we assume that an external tokenization
        # was used and we don't include the tokenizer in the exported graph.
        tokenizer = SpaceTokenizer(in_graph=False)
    return tokenizer


@register_tokenizer
class SpaceTokenizer(Tokenizer):
    """A tokenizer that splits on spaces."""

    def __init__(self, in_graph=True):
        """Initializes the tokenizer.

        Args:
          in_graph: If ``True``, this tokenizer should be integrated in the exported graph.
        """
        self._in_graph = in_graph

    @property
    def in_graph(self):
        return self._in_graph

    def _tokenize_tensor(self, text):
        return self._tokenize_batch_tensor(text)

    def _tokenize_batch_tensor(self, text):
        return tf.strings.split(text)

    def _detokenize_tensor(self, tokens):
        return self._detokenize_ragged_tensor(tokens)

    def _detokenize_batch_tensor(self, tokens, sequence_length):
        ragged = tf.RaggedTensor.from_tensor(tokens, lengths=sequence_length)
        return self._detokenize_ragged_tensor(ragged)

    def _detokenize_ragged_tensor(self, tokens):
        return tf.strings.reduce_join(tokens, axis=tokens.shape.rank - 1, separator=" ")

    def _tokenize_string(self, text):
        return text.split()

    def _detokenize_string(self, tokens):
        return " ".join(tokens)


@register_tokenizer
class CharacterTokenizer(Tokenizer):
    """A tokenizer that splits unicode characters."""

    @property
    def in_graph(self):
        return True

    def _tokenize_tensor(self, text):
        return self._tokenize_batch_tensor(text)

    def _tokenize_batch_tensor(self, text):
        text = tf.strings.regex_replace(text, " ", "▁")
        return tf.strings.unicode_split(text, "UTF-8")

    def _detokenize_tensor(self, tokens):
        return self._detokenize_ragged_tensor(tokens)

    def _detokenize_batch_tensor(self, tokens, sequence_length):
        _ = sequence_length
        return self._detokenize_ragged_tensor(tokens)

    def _detokenize_ragged_tensor(self, tokens):
        text = tf.strings.reduce_join(tokens, axis=tokens.shape.rank - 1)
        return tf.strings.regex_replace(text, "▁", " ")

    def _tokenize_string(self, text):
        return list(text.replace(" ", "▁"))

    def _detokenize_string(self, tokens):
        return "".join(tokens).replace("▁", " ")
