"""SentencePiece tokenizer."""

import tensorflow as tf
import tensorflow_text as tft

from opennmt.tokenizers import tokenizer


@tokenizer.register_tokenizer
class SentencePieceTokenizer(tokenizer.TensorFlowTokenizer):
    """In-graph SentencePiece tokenizer using
    ``tensorflow_text.SentencepieceTokenizer``.
    """

    def __init__(self, model, nbest_size=0, alpha=1.0):
        """Initializes the tokenizer.

        Args:
          model: Path to the SentencePiece model.
          nbest_size: Number of candidates to sample from (disabled during inference).
          alpha: Smoothing parameter for the sampling.
        """
        super().__init__()
        self._nbest_size = nbest_size
        with tf.io.gfile.GFile(model, "rb") as model_file:
            self._tokenizer = tft.SentencepieceTokenizer(
                model=model_file.read(),
                out_type=tf.string,
                nbest_size=nbest_size,
                alpha=alpha,
            )

    def _tokenize_tensor(self, text, training):
        if not training:
            self._tokenizer.nbest_size = 0
        tokens = self._tokenizer.tokenize(text)
        if not training:
            self._tokenizer.nbest_size = self._nbest_size
        return tokens

    def _detokenize_tensor(self, tokens):
        return self._tokenizer.detokenize(tokens)
