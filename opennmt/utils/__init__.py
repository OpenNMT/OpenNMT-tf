"""Module defining various utilities."""

from opennmt.utils.checkpoint import (
    average_checkpoints,
    average_checkpoints_into_layer,
    is_v1_checkpoint,
)
from opennmt.utils.decoding import (
    BeamSearch,
    BestSampler,
    DecodingResult,
    DecodingStrategy,
    GreedySearch,
    RandomSampler,
    Sampler,
    dynamic_decode,
)
from opennmt.utils.exporters import (
    CheckpointExporter,
    CTranslate2Exporter,
    CTranslate2Float16Exporter,
    CTranslate2Int8Exporter,
    CTranslate2Int8Float16Exporter,
    CTranslate2Int16Exporter,
    Exporter,
    SavedModelExporter,
    TFLiteExporter,
    TFLiteFloat16Exporter,
    register_exporter,
)
from opennmt.utils.losses import (
    cross_entropy_loss,
    cross_entropy_sequence_loss,
    guided_alignment_cost,
    max_margin_loss,
    regularization_penalty,
)
from opennmt.utils.misc import format_translation_output
from opennmt.utils.scorers import (
    BLEUScorer,
    PRFScorer,
    ROUGEScorer,
    Scorer,
    TERScorer,
    WERScorer,
    make_scorers,
    register_scorer,
)
from opennmt.utils.tensor import roll_sequence
