"""Module defining decoders."""

from opennmt.decoders.decoder import Decoder, get_sampling_probability
from opennmt.decoders.rnn_decoder import (
    AttentionalRNNDecoder,
    RNMTPlusDecoder,
    RNNDecoder,
)
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
