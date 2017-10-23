"""Module defining various utilities."""

from opennmt.utils.reducer import SumReducer
from opennmt.utils.reducer import MultiplyReducer
from opennmt.utils.reducer import ConcatReducer
from opennmt.utils.reducer import JoinReducer

from opennmt.utils.bridge import CopyBridge
from opennmt.utils.bridge import ZeroBridge
from opennmt.utils.bridge import DenseBridge

from opennmt.utils.position import PositionEmbedder

from opennmt.utils.vocab import Vocab
