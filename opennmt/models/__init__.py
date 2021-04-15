"""Module defining models."""

from opennmt.models.catalog import GPT2Small
from opennmt.models.catalog import ListenAttendSpell
from opennmt.models.catalog import LstmCnnCrfTagger
from opennmt.models.catalog import LuongAttention
from opennmt.models.catalog import TransformerBase
from opennmt.models.catalog import TransformerBaseRelative
from opennmt.models.catalog import TransformerBaseSharedEmbeddings
from opennmt.models.catalog import TransformerBig
from opennmt.models.catalog import TransformerBigRelative
from opennmt.models.catalog import TransformerBigSharedEmbeddings
from opennmt.models.catalog import get_model_from_catalog
from opennmt.models.catalog import register_model_in_catalog

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
