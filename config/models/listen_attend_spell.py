"""Defines a model similar to the "Listen, Attend and Spell" model described
in https://arxiv.org/abs/1508.01211.
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.SequenceRecordInputter(
          input_depth_key="input_depth"),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_vocabulary",
          embedding_size=50),
      encoder=onmt.encoders.PyramidalRNNEncoder(
          num_layers=3,
          num_units=512,
          reduction_factor=2,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3),
      decoder=onmt.decoders.MultiAttentionalRNNDecoder(
          num_layers=3,
          num_units=512,
          attention_layers=[0],
          attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
          cell_class=tf.contrib.rnn.LSTMCell,
          dropout=0.3,
          residual_connections=False))
