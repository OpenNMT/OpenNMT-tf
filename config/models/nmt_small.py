"""Defines a small unidirectional LSTM encoder-decoder model."""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_words_vocabulary",
          embedding_size=512),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_words_vocabulary",
          embedding_size=512),
      encoder=onmt.encoders.UnidirectionalRNNEncoder(
          num_layers=2,
          num_units=512,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=2,
          num_units=512,
          bridge=onmt.layers.CopyBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False))
