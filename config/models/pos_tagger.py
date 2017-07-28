import tensorflow as tf
import opennmt as onmt

def model():
  return onmt.models.SequenceTagger(
    embedder=onmt.embedders.WordEmbedder(
      vocabulary_file="data/en-dict.txt",
      embedding_size=80),
    encoder=onmt.encoders.BidirectionalRNNEncoder(
      num_layers=2,
      num_units=128,
      cell_class=tf.contrib.rnn.LSTMCell,
      dropout=0.3,
      residual_connections=False),
    labels_vocabulary_file="data/wsj/tags.txt")

def train(model):
  model.set_filters(maximum_length=70)

def infer(model):
  pass
