"""Module defining models."""

from opennmt.models.catalog import GPT2Small
from opennmt.models.catalog import ListenAttendSpell
from opennmt.models.catalog import LstmCnnCrfTagger
from opennmt.models.catalog import LuongAttention
from opennmt.models.catalog import Transformer as TransformerBase
from opennmt.models.catalog import TransformerBig

from opennmt.models.language_model import LanguageModel
from opennmt.models.language_model import LanguageModelInputter

from opennmt.models.model import Model
from opennmt.models.model import SequenceGenerator

from opennmt.models.sequence_classifier import ClassInputter
from opennmt.models.sequence_classifier import SequenceClassifier

from opennmt.models.sequence_tagger import SequenceTagger
from opennmt.models.sequence_tagger import TagsInputter

from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel
from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.models.sequence_to_sequence import SequenceToSequenceInputter

from opennmt.models.transformer import Transformer
