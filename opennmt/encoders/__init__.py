"""Module defining encoders."""

from opennmt.encoders.encoder import SequentialEncoder, ParallelEncoder
from opennmt.encoders.rnn_encoder import UnidirectionalRNNEncoder
from opennmt.encoders.rnn_encoder import BidirectionalRNNEncoder
from opennmt.encoders.rnn_encoder import RNMTPlusEncoder
from opennmt.encoders.rnn_encoder import GoogleRNNEncoder
from opennmt.encoders.rnn_encoder import PyramidalRNNEncoder
from opennmt.encoders.conv_encoder import ConvEncoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.encoders.mean_encoder import MeanEncoder
