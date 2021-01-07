"""Module defining tokenizers.

Tokenizers can work on string ``tf.Tensor`` as in-graph transformation.
"""

try:
    import pyonmttok
    from opennmt.tokenizers.opennmt_tokenizer import OpenNMTTokenizer
except ImportError:
    pass

from opennmt.tokenizers.tokenizer import Tokenizer
from opennmt.tokenizers.tokenizer import SpaceTokenizer
from opennmt.tokenizers.tokenizer import CharacterTokenizer
from opennmt.tokenizers.tokenizer import make_tokenizer
from opennmt.tokenizers.tokenizer import register_tokenizer
