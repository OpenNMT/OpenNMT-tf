"""Module defining reusable and model specific layers."""

from opennmt.layers.bridge import Bridge
from opennmt.layers.bridge import CopyBridge
from opennmt.layers.bridge import DenseBridge
from opennmt.layers.bridge import ZeroBridge

from opennmt.layers.common import Dense
from opennmt.layers.common import LayerNorm
from opennmt.layers.common import LayerWrapper
from opennmt.layers.common import dropout
from opennmt.layers.common import gelu

from opennmt.layers.position import PositionEmbedder
from opennmt.layers.position import PositionEncoder
from opennmt.layers.position import SinusoidalPositionEncoder

from opennmt.layers.reducer import ConcatReducer
from opennmt.layers.reducer import DenseReducer
from opennmt.layers.reducer import JoinReducer
from opennmt.layers.reducer import MultiplyReducer
from opennmt.layers.reducer import Reducer
from opennmt.layers.reducer import SumReducer

from opennmt.layers.rnn import LSTM
from opennmt.layers.rnn import RNN
from opennmt.layers.rnn import RNNCellWrapper
from opennmt.layers.rnn import make_rnn_cell

from opennmt.layers.transformer import FeedForwardNetwork
from opennmt.layers.transformer import MultiHeadAttention
from opennmt.layers.transformer import MultiHeadAttentionReduction
from opennmt.layers.transformer import SelfAttentionDecoderLayer
from opennmt.layers.transformer import SelfAttentionEncoderLayer
from opennmt.layers.transformer import TransformerLayerWrapper
from opennmt.layers.transformer import combine_heads
from opennmt.layers.transformer import future_mask
from opennmt.layers.transformer import split_heads
