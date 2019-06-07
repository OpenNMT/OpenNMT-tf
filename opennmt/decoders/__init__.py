"""Module defining decoders."""

from opennmt.decoders.decoder import Decoder
from opennmt.decoders.decoder import get_sampling_probability

from opennmt.decoders.rnn_decoder import AttentionalRNNDecoder
from opennmt.decoders.rnn_decoder import RNMTPlusDecoder
from opennmt.decoders.rnn_decoder import RNNDecoder

from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
