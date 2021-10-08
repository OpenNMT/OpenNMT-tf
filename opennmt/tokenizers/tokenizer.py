"""Define base tokenizers."""

import abc
import json
import sys

import tensorflow as tf
import yaml

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
        self,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        delimiter=" ",
        training=True,
    ):
        """Tokenizes a stream of sentences.

        Args:
          input_stream: The input stream.
          output_stream: The output stream.
          delimiter: The token delimiter to use for text serialization.
          training: Set to ``False`` to tokenize for inference.
        """
        for line in input_stream:
            line = line.strip()
            tokens = self.tokenize(line, training=training)
            merged_tokens = delimiter.join(tokens)
            misc.print_as_bytes(merged_tokens, stream=output_stream)

    def detokenize_stream(
        self,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        delimiter=" ",
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

    def tokenize(self, text, training=True):
        """Tokenizes text.

        Args:
          text: A string or batch of strings to tokenize as a ``tf.Tensor`` or
            Python values.
          training: Set to ``False`` to tokenize for inference.

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
            return self._tokenize(text, training)

    def _tokenize(self, text, training):
        if tf.is_tensor(text):
            return self._tokenize_tensor(text, training)
        elif isinstance(text, list):
            return list(map(lambda t: self._tokenize(t, training), text))
        else:
            text = tf.compat.as_text(text)
            return self._tokenize_string(text, training)

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
            return self._detokenize_tensor(tokens)
        elif tf.is_tensor(tokens):
            rank = len(tokens.shape)
            if rank == 1:
                return self._detokenize_tensor(tokens)
            elif rank == 2:
                if sequence_length is None:
                    raise ValueError(
                        "sequence_length is required for Tensor detokenization"
                    )
                tokens = tf.RaggedTensor.from_tensor(tokens, lengths=sequence_length)
                return self._detokenize_tensor(tokens)
            else:
                raise ValueError("Unsupported tensor rank %d for detokenization" % rank)
        elif isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
            return list(map(self.detokenize, tokens))
        else:
            tokens = [tf.compat.as_text(token) for token in tokens]
            return self._detokenize_string(tokens)

    def _tokenize_tensor(self, text, training):
        """Tokenizes a tensor.

        When not overriden, this default implementation calls the string-based
        tokenization.

        Args:
          text: A 0-D or 1-D string ``tf.Tensor``.
          training: Set to ``False`` to tokenize for inference.

        Returns:
          A 1-D string ``tf.Tensor``, or a 2-D string ``tf.RaggedTensor`` if the
          input was a batch of text.
        """

        def _python_wrapper(string_t):
            string = tf.compat.as_text(string_t.numpy())
            tokens = self._tokenize_string(string, training)
            return tf.constant(tokens, dtype=tf.string)

        def _python_wrapper_batch(batch_text):
            batch_text = list(map(tf.compat.as_text, batch_text.numpy()))
            batch_tokens = self._tokenize_string_batch(batch_text, training)
            flat_tokens = tf.constant(tf.nest.flatten(batch_tokens), dtype=tf.string)
            lengths = tf.constant(list(map(len, batch_tokens)), dtype=tf.int32)
            return flat_tokens, lengths

        rank = text.shape.rank
        if rank == 0:
            tokens = tf.py_function(_python_wrapper, [text], tf.string)
            tokens.set_shape([None])
            return tokens
        elif rank == 1:
            flat_tokens, lengths = tf.py_function(
                _python_wrapper_batch, [text], (tf.string, tf.int32)
            )
            flat_tokens.set_shape([None])
            lengths.set_shape([None])
            return tf.RaggedTensor.from_row_lengths(flat_tokens, lengths)
        else:
            raise ValueError("Unsupported tensor rank %d for tokenization" % rank)

    def _detokenize_tensor(self, tokens):
        """Detokenizes tokens.

        When not overriden, this default implementation calls the string-based
        detokenization.

        Args:
          tokens: A 1-D string ``tf.Tensor``, or a 2-D string ``tf.RaggedTensor``.

        Returns:
          A 0-D string ``tf.Tensor``, or a 1-D string ``tf.Tensor`` if the input
          was a batch of tokens.
        """

        def _python_wrapper(tokens_t):
            tokens = [tf.compat.as_text(s) for s in tokens_t.numpy()]
            string = self._detokenize_string(tokens)
            return tf.constant(string)

        rank = tokens.shape.rank
        if rank == 1:
            text = tf.py_function(_python_wrapper, [tokens], tf.string)
            text.set_shape([])
            return text
        elif rank == 2:
            return tf.map_fn(self._detokenize_tensor, tokens, dtype=tf.string)
        else:
            raise ValueError("Unsupported tensor rank %d for detokenization" % rank)

    @abc.abstractmethod
    def _tokenize_string(self, text, training):
        """Tokenizes a Python unicode string.

        This method should be thread-safe.

        Args:
          text: A Python unicode string.
          training: Set to ``False`` to tokenize for inference.

        Returns:
          A list of Python unicode strings.
        """
        raise NotImplementedError()

    def _tokenize_string_batch(self, batch_text, training):
        """Tokenizes a batch of Python unicode strings.

        Args:
          batch_text: A list of Python unicode strings.
          training: Set to ``False`` to tokenize for inference.

        Returns:
          A list of lists of Python unicode strings.
        """
        return [self._tokenize_string(text, training) for text in batch_text]

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


def _process_stream_as_dataset(
    input_stream,
    output_stream,
    map_func,
    batch_size=512,
    num_parallel_calls=4,
):
    dataset = tf.data.Dataset.from_generator(
        lambda: input_stream,
        output_types=tf.string,
        output_shapes=tf.TensorShape([]),
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(map_func, num_parallel_calls=num_parallel_calls)

    expected_spec = tf.TensorSpec(shape=[None], dtype=tf.string)
    if dataset.element_spec != expected_spec:
        raise TypeError(
            "Expected map_func to produce elements with spec %s, but got spec %s instead"
            % (expected_spec, dataset.element_spec)
        )

    for lines in dataset.as_numpy_iterator():
        for line in lines:
            misc.print_as_bytes(line, stream=output_stream)


class TensorFlowTokenizer(tf.Module, Tokenizer):
    """Base class for tokenizers using only TensorFlow ops."""

    @property
    def in_graph(self):
        return True

    def tokenize_stream(
        self,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        delimiter=" ",
        training=True,
    ):
        def _map_func(line):
            line = tf.strings.strip(line)
            tokens = self._tokenize_tensor(line, training)
            return tf.strings.reduce_join(
                tokens, axis=tokens.shape.rank - 1, separator=delimiter
            )

        _process_stream_as_dataset(input_stream, output_stream, _map_func)

    def detokenize_stream(
        self,
        input_stream=sys.stdin,
        output_stream=sys.stdout,
        delimiter=" ",
    ):
        def _map_func(line):
            line = tf.strings.strip(line)
            tokens = tf.strings.split(line, sep=delimiter)
            return self._detokenize_tensor(tokens)

        _process_stream_as_dataset(input_stream, output_stream, _map_func)

    @abc.abstractmethod
    def _tokenize_tensor(self, text, training):
        raise NotImplementedError()

    @abc.abstractmethod
    def _detokenize_tensor(self, tokens):
        raise NotImplementedError()

    def _tokenize_string(self, text, training):
        tokens = self._tokenize_tensor(tf.constant(text, dtype=tf.string), training)
        return [token.decode("utf-8") for token in tokens.numpy()]

    def _detokenize_string(self, tokens):
        text = self._detokenize_tensor(tf.constant(tokens, dtype=tf.string))
        return text.numpy().decode("utf-8")


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

    def _tokenize_tensor(self, text, _):
        return tf.strings.split(text)

    def _detokenize_tensor(self, tokens):
        return tf.strings.reduce_join(tokens, axis=tokens.shape.rank - 1, separator=" ")

    def _tokenize_string(self, text, _):
        return text.split()

    def _detokenize_string(self, tokens):
        return " ".join(tokens)


@register_tokenizer
class CharacterTokenizer(Tokenizer):
    """A tokenizer that splits unicode characters."""

    @property
    def in_graph(self):
        return True

    def _tokenize_tensor(self, text, _):
        text = tf.strings.regex_replace(text, " ", "▁")
        return tf.strings.unicode_split(text, "UTF-8")

    def _detokenize_tensor(self, tokens):
        text = tf.strings.reduce_join(tokens, axis=tokens.shape.rank - 1)
        return tf.strings.regex_replace(text, "▁", " ")

    def _tokenize_string(self, text, _):
        return list(text.replace(" ", "▁"))

    def _detokenize_string(self, tokens):
        return "".join(tokens).replace("▁", " ")
