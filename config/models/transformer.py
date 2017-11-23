"""Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",
          embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1)
