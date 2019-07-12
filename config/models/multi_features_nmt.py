"""Defines a sequence to sequence model with multiple input features. For
example, this could be words, parts of speech, and lemmas that are embedded in
parallel and concatenated into a single input embedding.

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
import tensorflow_addons as tfa
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(embedding_size=512),
          onmt.inputters.WordEmbedder(embedding_size=16),
          onmt.inputters.WordEmbedder(embedding_size=64)],
          reducer=onmt.layers.ConcatReducer()),
      target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
      encoder=onmt.encoders.RNNEncoder(
          num_layers=4,
          num_units=256,
          bidrectional=True,
          dropout=0.3,
          residual_connections=False,
          reducer=onmt.layers.ConcatReducer(),
          cell_class=tf.keras.layers.LSTMCell),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=4,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tfa.seq2seq.LuongAttention,
          cell_class=tf.keras.layers.LSTMCell,
          dropout=0.3,
          residual_connections=False))
