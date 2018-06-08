"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt


class ListenAttendSpell(onmt.models.SequenceToSequence):
  """Defines a model similar to the "Listen, Attend and Spell" model described
  in https://arxiv.org/abs/1508.01211.
  """
  def __init__(self):
    super(ListenAttendSpell, self).__init__(
        source_inputter=onmt.inputters.SequenceRecordInputter(),
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

class NMTBig(onmt.models.SequenceToSequence):
  """Defines a bidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTBig, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=4,
            num_units=1024,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=1024,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTMedium(onmt.models.SequenceToSequence):
  """Defines a medium-sized bidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTMedium, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
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

class NMTSmall(onmt.models.SequenceToSequence):
  """Defines a small unidirectional LSTM encoder-decoder model."""
  def __init__(self):
    super(NMTSmall, self).__init__(
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

class SeqTagger(onmt.models.SequenceTagger):
  """Defines a bidirectional LSTM-CNNs-CRF as described in https://arxiv.org/abs/1603.01354."""
  def __init__(self):
    # pylint: disable=bad-continuation
    super(SeqTagger, self).__init__(
        inputter=onmt.inputters.MixedInputter([
            onmt.inputters.WordEmbedder(
                vocabulary_file_key="words_vocabulary",
                embedding_size=None,
                embedding_file_key="words_embedding",
                trainable=True),
            onmt.inputters.CharConvEmbedder(
                vocabulary_file_key="chars_vocabulary",
                embedding_size=30,
                num_outputs=30,
                kernel_size=3,
                stride=1,
                dropout=0.5)],
            dropout=0.5),
        encoder=onmt.encoders.BidirectionalRNNEncoder(
            num_layers=1,
            num_units=400,
            reducer=onmt.layers.ConcatReducer(),
            cell_class=tf.contrib.rnn.LSTMCell,
            dropout=0.5,
            residual_connections=False),
        labels_vocabulary_file_key="tags_vocabulary",
        crf_decoding=True)

class Transformer(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self):
    super(Transformer, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1)

class TransformerAAN(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762
  with cumulative average attention in the decoder as described in
  https://arxiv.org/abs/1805.00631."""
  def __init__(self):
    super(TransformerAAN, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,
        decoder_self_attention_type="average")

class TransformerBig(onmt.models.Transformer):
  """Defines a large Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self):
    super(TransformerBig, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=1024),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=1024),
        num_layers=6,
        num_units=1024,
        num_heads=16,
        ffn_inner_dim=4096,
        dropout=0.3,
        attention_dropout=0.1,
        relu_dropout=0.1)
