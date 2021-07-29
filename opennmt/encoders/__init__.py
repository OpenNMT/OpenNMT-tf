"""Module defining encoders."""

from opennmt.encoders.conv_encoder import ConvEncoder
from opennmt.encoders.encoder import Encoder, ParallelEncoder, SequentialEncoder
from opennmt.encoders.mean_encoder import MeanEncoder
from opennmt.encoders.rnn_encoder import (
    GNMTEncoder,
    LSTMEncoder,
    PyramidalRNNEncoder,
    RNMTPlusEncoder,
    RNNEncoder,
)
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
