"""Module defining reusable and model specific layers."""

from opennmt.layers.common import LayerNorm, LayerWrapper

from opennmt.layers.reducer import SumReducer
from opennmt.layers.reducer import MultiplyReducer
from opennmt.layers.reducer import ConcatReducer
from opennmt.layers.reducer import JoinReducer

from opennmt.layers.bridge import CopyBridge
from opennmt.layers.bridge import ZeroBridge
from opennmt.layers.bridge import DenseBridge

from opennmt.layers.position import PositionEmbedder
from opennmt.layers.position import SinusoidalPositionEncoder

from opennmt.layers.rnn import make_rnn_cell, RNNCellWrapper, RNN

from opennmt.layers.transformer import FeedForwardNetwork, MultiHeadAttention
