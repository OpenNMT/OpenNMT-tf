"""Defines a Transformer model with multiple input features. For example, these
could be words, parts of speech, and lemmas that are embedded in parallel and
concatenated into a single input embedding.

The features are separate data files with separate vocabularies. The YAML
configuration file should look like this:

data:
  train_features_file:
    - features_1.txt
    - features_2.txt
    - features_3.txt
  train_labels_file: target.txt
  source_1_vocabulary: feature_1_vocab.txt
  source_2_vocabulary: feature_2_vocab.txt
  source_3_vocabulary: feature_3_vocab.txt
  target_vocabulary: target_vocab.txt
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.Transformer(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(embedding_size=512),
          onmt.inputters.WordEmbedder(embedding_size=16),
          onmt.inputters.WordEmbedder(embedding_size=64)],
          reducer=onmt.layers.ConcatReducer()),
      target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      num_layers=6,
      num_units=512,
      num_heads=8,
      ffn_inner_dim=2048,
      dropout=0.1,
      attention_dropout=0.1,
      ffn_dropout=0.1)
