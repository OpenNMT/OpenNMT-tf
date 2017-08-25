"""Defines a "Bi-directional LSTM-CNNs-CRF" as described in
https://arxiv.org/pdf/1603.01354.pdf.
"""

import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceTagger(
    embedder=onmt.embedders.MixedEmbedder([
      onmt.embedders.WordEmbedder(
        vocabulary_file_key="words_vocabulary",
        embedding_size=None,
        embedding_file_key="words_embedding",
        trainable=True),
      onmt.embedders.CharConvEmbedder(
        vocabulary_file_key="chars_vocabulary",
        embedding_size=30,
        num_outputs=30,
        kernel_size=3,
        stride=1,
        dropout=0.5)],
      dropout=0.5),
    encoder=onmt.encoders.BidirectionalRNNEncoder(
      num_layers=1,
      num_units=200,
      cell_class=tf.contrib.rnn.LSTMCell,
      dropout=0.5,
      residual_connections=False),
    labels_vocabulary_file_key="tags_vocabulary",
    crf_decoding=True)
