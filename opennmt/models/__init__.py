"""Module defining models."""

from opennmt.models.language_model import LanguageModel
from opennmt.models.language_model import LanguageModelInputter

from opennmt.models.sequence_classifier import ClassInputter
from opennmt.models.sequence_classifier import SequenceClassifier

from opennmt.models.sequence_tagger import SequenceTagger
from opennmt.models.sequence_tagger import TagsInputter

from opennmt.models.sequence_to_sequence import EmbeddingsSharingLevel
from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.models.sequence_to_sequence import SequenceToSequenceInputter

from opennmt.models.transformer import Transformer
