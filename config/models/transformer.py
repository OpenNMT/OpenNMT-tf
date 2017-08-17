"""Implements a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
    source_embedder=onmt.embedders.WordEmbedder(
      vocabulary_file="data/en-dict.txt",
      embedding_size=512),
    target_embedder=onmt.embedders.WordEmbedder(
      vocabulary_file="data/fr-dict.txt",
      embedding_size=512),
    num_layers=4,
    num_heads=8,
    ffn_inner_dim=2048,
    dropout=0.1)
