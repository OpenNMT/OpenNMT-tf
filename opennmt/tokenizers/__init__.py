"""Module defining tokenizers.

Tokenizers can work on string ``tf.Tensor`` as in-graph transformation.
"""

try:
    import pyonmttok

    from opennmt.tokenizers.opennmt_tokenizer import OpenNMTTokenizer
except ImportError:
    pass

try:
    from opennmt.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer
except ImportError:
    pass

from opennmt.tokenizers.tokenizer import (
    CharacterTokenizer,
    SpaceTokenizer,
    Tokenizer,
    make_tokenizer,
    register_tokenizer,
)
