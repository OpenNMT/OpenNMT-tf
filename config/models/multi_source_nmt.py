"""Defines a multi source sequence to sequence model. Source sequences are read
from 2 files, encoded separately, and the encoder outputs are concatenated in
time.
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
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
      encoder=onmt.encoders.ParallelEncoder([
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.layers.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False),
          onmt.encoders.BidirectionalRNNEncoder(
              num_layers=2,
              num_units=512,
              reducer=onmt.layers.ConcatReducer(),
              cell_class=tf.contrib.rnn.LSTMCell,
              dropout=0.3,
              residual_connections=False)],
          outputs_reducer=onmt.layers.ConcatReducer(axis=1)),
      decoder=onmt.decoders.AttentionalRNNDecoder(
          num_layers=4,
          num_units=512,
          bridge=onmt.layers.DenseBridge(),
          attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False))
