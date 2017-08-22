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
               source_embedder,
               target_embedder,
               num_layers,
               num_heads,
               ffn_inner_dim,
               dropout=0.1,
               position_encoder=PositionEmbedder(),
               name="transformer"):
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
      source_embedder,
      target_embedder,
      encoder,
      decoder,
      name=name)
