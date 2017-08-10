"""Defines a "Bi-directional LSTM-CNNs-CRF" as described in
https://arxiv.org/pdf/1603.01354.pdf.
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceTagger(
    embedder=onmt.embedders.MixedEmbedder([
      onmt.embedders.WordEmbedder(
        vocabulary_file="data/en-dict.txt",
        embedding_size=None,
        embedding_file="data/glove.6B.50d.txt",
        trainable=True),
      onmt.embedders.CharConvEmbedder(
        vocabulary_file="data/en-char-dict.txt",
        embedding_size=20,
        num_outputs=20,
        kernel_size=5,
        stride=3,
        dropout=0.2)]),
    encoder=onmt.encoders.BidirectionalRNNEncoder(
      num_layers=2,
      num_units=128,
      cell_class=tf.contrib.rnn.LSTMCell,
      dropout=0.3,
      residual_connections=False),
    labels_vocabulary_file="data/wsj/tags.txt",
    crf_decoding=True)

def train(model):
  model.set_filters(maximum_length=70)

def infer(model):
  pass
