"""Defines a character-based sequence-to-sequence model.

Character vocabularies can be built with:

python -m bin.build_vocab --tokenizer CharacterTokenizer ...
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceToSequence(
      source_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="source_chars_vocabulary",
          embedding_size=30,
          tokenizer=onmt.tokenizers.CharacterTokenizer()),
      target_inputter=onmt.inputters.WordEmbedder(
          vocabulary_file_key="target_chars_vocabulary",
          embedding_size=30,
          tokenizer=onmt.tokenizers.CharacterTokenizer()),
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
