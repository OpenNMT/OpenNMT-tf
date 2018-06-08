"""Define the Google's Transformer model."""

import tensorflow as tf

from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.encoders.lnmasru_encoder import LNMASRUBiEncoder
from opennmt.decoders.lnmasru_decoder import LNMASRUDecoder
from opennmt.encoders import UnidirectionalRNNEncoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
import opennmt.layers.bridge


class TransLNMASRU(SequenceToSequence):
  """Attention-based sequence-to-sequence model as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               source_inputter,
               target_inputter,
               num_layers,
               num_units,
               num_heads,
               attention_dropout,
               dropout=0.1,
               name="lnmasru"):
    """Initializes a Transformer model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the source data.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputter` to process
        the target data. Currently, only the
        :class:`opennmt.inputters.text_inputter.WordEmbedder` is supported.
      num_layers: The shared number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in each self-attention layers.
      ffn_inner_dim: The inner dimension of the feed forward layers.
      dropout: The probability to drop units in each layer output.
      attention_dropout: The probability to drop units from the attention.
      relu_dropout: The probability to drop units from the ReLU activation in
        the feed forward layer.
      position_encoder: A :class:`opennmt.layers.position.PositionEncoder` to
        apply on the inputs.
      name: The name of this model.
    """
    #encoder = LNMASRUBiEncoder(
    #    num_layers,
    #    num_units=num_units,
    #    dropout=dropout)
    encoder=SelfAttentionEncoder(
        num_layers,
        num_units=num_units,
        num_heads=num_heads,
        ffn_inner_dim=2048,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=dropout,
        position_encoder=SinusoidalPositionEncoder())
    decoder = LNMASRUDecoder(
        num_layers,
        num_units=num_units,
        num_heads=num_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        bridge=opennmt.layers.bridge.ZeroBridge)

    super(TransLNMASRU, self).__init__(
        source_inputter,
        target_inputter,
        encoder,
        decoder,
        daisy_chain_variables=False,
        name=name)

  def _initializer(self, params):
    return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)

class LNMASRUTrans(SequenceToSequence):
  """Attention-based sequence-to-sequence model as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               source_inputter,
               target_inputter,
               num_layers,
               num_units,
               num_heads,
               attention_dropout,
               dropout=0.1,
               name="lnmasru"):
    """Initializes a Transformer model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputt
        the source data.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputt
        the target data. Currently, only the
        :class:`opennmt.inputters.text_inputter.WordEmbedder` is s
      num_layers: The shared number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in each self-attention layers
      ffn_inner_dim: The inner dimension of the feed forward layer
      dropout: The probability to drop units in each layer output.
      attention_dropout: The probability to drop units from the at
      relu_dropout: The probability to drop units from the ReLU ac
        the feed forward layer.
      position_encoder: A :class:`opennmt.layers.position.Position
        apply on the inputs.
      name: The name of this model.
    """
    encoder = LNMASRUBiEncoder(
                num_layers,
                num_units=num_units,
                dropout=dropout)
    decoder=SelfAttentionDecoder(
        num_layers,
        num_units=num_units,
        num_heads=num_heads,
        ffn_inner_dim=2048,
        dropout=dropout,
        attention_dropout=attention_dropout,
        relu_dropout=dropout,
        position_encoder=SinusoidalPositionEncoder())

    super(LNMASRUTrans, self).__init__(
        source_inputter,
        target_inputter,
        encoder,
        decoder,
        daisy_chain_variables=False,
        name=name)

  def _initializer(self, params):
    return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)


class LNMASRU(SequenceToSequence):
  """Attention-based sequence-to-sequence model as described in
  https://arxiv.org/abs/1706.03762.
  """
  def __init__(self,
              source_inputter,
              target_inputter,
              num_layers,
              num_units,
              num_heads,
              attention_dropout,
              dropout=0.1,
              name="lnmasru"):
    """Initializes a Transformer model.

    Args:
      source_inputter: A :class:`opennmt.inputters.inputter.Inputt
        the source data.
      target_inputter: A :class:`opennmt.inputters.inputter.Inputt
        the target data. Currently, only the
        :class:`opennmt.inputters.text_inputter.WordEmbedder` is s
      num_layers: The shared number of layers.
      num_units: The number of hidden units.
      num_heads: The number of heads in each self-attention layers
      ffn_inner_dim: The inner dimension of the feed forward layer
      dropout: The probability to drop units in each layer output.
      attention_dropout: The probability to drop units from the at
      relu_dropout: The probability to drop units from the ReLU ac
        the feed forward layer.
      position_encoder: A :class:`opennmt.layers.position.Position
        apply on the inputs.
      name: The name of this model.
    """
    encoder = LNMASRUBiEncoder(
                num_layers,
                num_units=num_units,
                dropout=dropout)
    decoder = LNMASRUDecoder(
                num_layers,
                num_units=num_units,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                bridge=opennmt.layers.bridge.ZeroBridge)
    super(LNMASRU, self).__init__(
        source_inputter,
        target_inputter,
        encoder,
        decoder,
        daisy_chain_variables=False,
        name=name)

  def _initializer(self, params):
    return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)
