"""Define the Google's Transformer model."""

from opennmt.models.sequence_to_sequence import SequenceToSequence
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.utils.position import PositionEmbedder


class Transformer(SequenceToSequence):
  """Attention-based sequence-to-sequence model as described in
  https://arxiv.org/abs/1706.03762.
  """

  def __init__(self,
               source_inputter,
               target_inputter,
               num_layers,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               position_encoder=PositionEmbedder(),
               name="transformer"):
    """Initializes a Transformer model.

    Args:
      source_inputter: A `onmt.inputters.Inputter` to process the source data.
      target_inputter: A `onmt.inputters.Inputter` to process the target data.
        Currently, only the `onmt.inputters.WordEmbedder` is supported.
      num_layers: The shared number of layers.
      num_heads: The number of heads in each self-attention layers.
      ffn_inner_dim: The inner dimension of the feed forward layers.
      dropout: The probability to drop units in each layer output.
      position_encoder: A `onmt.utils.PositionEncoder` to apply on the inputs.
      name: The name of this model.
    """
    encoder = SelfAttentionEncoder(
        num_layers,
        num_heads=num_heads,
        ffn_inner_dim=ffn_inner_dim,
        dropout=dropout,
        keep_layers_output=True,
        position_encoder=position_encoder)
    decoder = SelfAttentionDecoder(
        num_layers,
        num_heads=num_heads,
        ffn_inner_dim=ffn_inner_dim,
        dropout=dropout,
        position_encoder=position_encoder)

    super(Transformer, self).__init__(
        source_inputter,
        target_inputter,
        encoder,
        decoder,
        name=name)
