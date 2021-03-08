"""Module defining various utilities."""

from opennmt.utils.checkpoint import average_checkpoints
from opennmt.utils.checkpoint import average_checkpoints_into_layer
from opennmt.utils.checkpoint import is_v1_checkpoint

from opennmt.utils.decoding import BeamSearch
from opennmt.utils.decoding import BestSampler
from opennmt.utils.decoding import DecodingResult
from opennmt.utils.decoding import DecodingStrategy
from opennmt.utils.decoding import GreedySearch
from opennmt.utils.decoding import RandomSampler
from opennmt.utils.decoding import Sampler
from opennmt.utils.decoding import dynamic_decode

from opennmt.utils.exporters import CTranslate2Exporter
from opennmt.utils.exporters import CTranslate2Float16Exporter
from opennmt.utils.exporters import CTranslate2Int16Exporter
from opennmt.utils.exporters import CTranslate2Int8Exporter
from opennmt.utils.exporters import CheckpointExporter
from opennmt.utils.exporters import Exporter
from opennmt.utils.exporters import SavedModelExporter
from opennmt.utils.exporters import TFLiteExporter
from opennmt.utils.exporters import TFLiteFloat16Exporter
from opennmt.utils.exporters import register_exporter

from opennmt.utils.losses import cross_entropy_loss
from opennmt.utils.losses import cross_entropy_sequence_loss
from opennmt.utils.losses import guided_alignment_cost
from opennmt.utils.losses import max_margin_loss
from opennmt.utils.losses import regularization_penalty

from opennmt.utils.misc import format_translation_output

from opennmt.utils.scorers import BLEUScorer
from opennmt.utils.scorers import PRFScorer
from opennmt.utils.scorers import ROUGEScorer
from opennmt.utils.scorers import Scorer
from opennmt.utils.scorers import TERScorer
from opennmt.utils.scorers import WERScorer
from opennmt.utils.scorers import make_scorers
from opennmt.utils.scorers import register_scorer

from opennmt.utils.tensor import roll_sequence
