"""Module defining models."""

from opennmt.models.catalog import (
    GPT2Small,
    ListenAttendSpell,
    LstmCnnCrfTagger,
    LuongAttention,
    ScalingNmtEnDe,
    ScalingNmtEnFr,
    TransformerBase,
    TransformerBaseRelative,
    TransformerBaseSharedEmbeddings,
    TransformerBig,
    TransformerBigRelative,
    TransformerBigSharedEmbeddings,
    TransformerTiny,
    get_model_from_catalog,
    register_model_in_catalog,
)
from opennmt.models.language_model import LanguageModel, LanguageModelInputter
from opennmt.models.model import Model, SequenceGenerator
from opennmt.models.sequence_classifier import ClassInputter, SequenceClassifier
from opennmt.models.sequence_tagger import SequenceTagger, TagsInputter
from opennmt.models.sequence_to_sequence import (
    EmbeddingsSharingLevel,
    SequenceToSequence,
    SequenceToSequenceInputter,
)
from opennmt.models.transformer import Transformer
