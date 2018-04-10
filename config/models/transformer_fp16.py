"""Defines a Transformer model with experimental FP16 data type."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",
          embedding_size=512,
          dtype=tf.float16),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512,
          dtype=tf.float16),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      relu_dropout=0.1)
