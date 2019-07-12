"""A Transformer model sharing all embeddings and softmax weights."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      ffn_dropout=0.1,
      share_embeddings=onmt.models.EmbeddingsSharingLevel.ALL)
