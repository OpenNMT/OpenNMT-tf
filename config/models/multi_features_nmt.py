"""Defines a sequence to sequence model with multiple input features. For
example, this could be words, parts of speech, and lemmas that are embedded in
parallel and concatenated into a single input embedding. The features are
separate data files with separate vocabularies.
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.ParallelInputter([
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="source_words_vocabulary",
              embedding_size=512),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="feature_1_vocabulary",
              embedding_size=16),
          onmt.inputters.WordEmbedder(
              vocabulary_file_key="feature_2_vocabulary",
              embedding_size=64)],
          reducer=onmt.layers.ConcatReducer()),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512),
      encoder=onmt.encoders.BidirectionalRNNEncoder(
          num_layers=4,
          num_units=512,
          reducer=onmt.layers.ConcatReducer(),
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=4,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False))
