"""Defines a dual source Transformer architecture with serial attention layers
and parameter sharing between the encoders.

See for example https://arxiv.org/pdf/1809.00188.pdf.
"""

import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_1",
              embedding_size=512),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_vocabulary_2",
              embedding_size=512)]),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_vocabulary",
          embedding_size=512),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1,
      share_encoders=True)
