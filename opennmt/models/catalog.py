"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt


class CharacterSeq2Seq(onmt.models.SequenceToSequence):
  """Defines a character-based sequence-to-sequence model.

  Character vocabularies can be built with:

  .. code-block:: text

      onmt-build-vocab --tokenizer CharacterTokenizer [...]
  """
  def __init__(self):
    super(CharacterSeq2Seq, self).__init__(
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

class MultiFeaturesNMT(onmt.models.SequenceToSequence):
  """Defines a sequence to sequence model with multiple input features. For
  example, this could be words, parts of speech, and lemmas that are embedded in
  parallel and concatenated into a single input embedding. The features are
  separate data files with separate vocabularies.
  """
  def __init__(self):
    # pylint: disable=bad-continuation
    super(MultiFeaturesNMT, self).__init__(
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

class MultiSourceNMT(onmt.models.SequenceToSequence):
  """Defines a multi source sequence to sequence model. Source sequences are read
  from 2 files, encoded separately, and the encoder outputs are concatenated in
  time.
  """
  def __init__(self):
    # pylint: disable=bad-continuation
    super(MultiSourceNMT, self).__init__(
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

class NMTMediumFP16(onmt.models.SequenceToSequence):
  """Defines a medium-sized bidirectional LSTM encoder-decoder model with
  experimental FP16 data type.
  """
  def __init__(self):
    super(NMTMediumFP16, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=tf.float16),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=tf.float16),
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

class TransformerFP16(onmt.models.Transformer):
  """Defines a Transformer model with experimental FP16 data type."""
  def __init__(self):
    super(TransformerFP16, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=tf.float16),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=tf.float16),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1)

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
