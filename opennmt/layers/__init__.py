"""Module defining reusable and model specific layers."""

from opennmt.layers.bridge import Bridge, CopyBridge, DenseBridge, ZeroBridge
from opennmt.layers.common import Dense, LayerNorm, LayerWrapper, dropout, gelu
from opennmt.layers.position import (
    PositionEmbedder,
    PositionEncoder,
    SinusoidalPositionEncoder,
)
from opennmt.layers.reducer import (
    ConcatReducer,
    DenseReducer,
    JoinReducer,
    MultiplyReducer,
    Reducer,
    SumReducer,
)
from opennmt.layers.rnn import LSTM, RNN, RNNCellWrapper, make_rnn_cell
from opennmt.layers.transformer import (
    FeedForwardNetwork,
    MultiHeadAttention,
    MultiHeadAttentionReduction,
    SelfAttentionDecoderLayer,
    SelfAttentionEncoderLayer,
    TransformerLayerWrapper,
    combine_heads,
    future_mask,
    split_heads,
)
