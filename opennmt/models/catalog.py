"""Catalog of predefined models."""

import tensorflow as tf
import opennmt as onmt

from opennmt.utils.misc import merge_dict


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
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3),
        decoder=onmt.decoders.MultiAttentionalRNNDecoder(
            num_layers=3,
            num_units=512,
            attention_layers=[0],
            attention_mechanism_class=tf.contrib.seq2seq.LuongMonotonicAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

  def auto_config(self, num_devices=1):
    config = super(ListenAttendSpell, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "GradientDescentOptimizer",
            "learning_rate": 0.2,
            "clip_gradients": 10.0,
            "scheduled_sampling_type": "constant",
            "scheduled_sampling_read_probability": 0.9
        },
        "train": {
            "batch_size": 32,
            "bucket_width": 15,
            "maximum_features_length": 2450,
            "maximum_labels_length": 330
        }
    })

class _RNNBase(onmt.models.SequenceToSequence):
  """Base class for RNN based NMT models."""
  def __init__(self, *args, **kwargs):
    super(_RNNBase, self).__init__(*args, **kwargs)

  def auto_config(self, num_devices=1):
    config = super(_RNNBase, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "AdamOptimizer",
            "learning_rate": 0.0002,
            "param_init": 0.1,
            "clip_gradients": 5.0
        },
        "train": {
            "batch_size": 64,
            "maximum_features_length": 80,
            "maximum_labels_length": 80
        }
    })

class NMTBig(_RNNBase):
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
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=1024,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTMedium(_RNNBase):
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
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=4,
            num_units=512,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False))

class NMTSmall(_RNNBase):
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
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.3,
            residual_connections=False),
        decoder=onmt.decoders.AttentionalRNNDecoder(
            num_layers=2,
            num_units=512,
            bridge=onmt.layers.CopyBridge(),
            attention_mechanism_class=tf.contrib.seq2seq.LuongAttention,
            cell_class=tf.nn.rnn_cell.LSTMCell,
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
            cell_class=tf.nn.rnn_cell.LSTMCell,
            dropout=0.5,
            residual_connections=False),
        labels_vocabulary_file_key="tags_vocabulary",
        crf_decoding=True)

  def auto_config(self, num_devices=1):
    config = super(SeqTagger, self).auto_config(num_devices=num_devices)
    return merge_dict(config, {
        "params": {
            "optimizer": "AdamOptimizer",
            "learning_rate": 0.001
        },
        "train": {
            "batch_size": 32
        }
    })

class Transformer(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32):
    super(Transformer, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1)

class TransformerFP16(Transformer):
  """Defines a Transformer model that uses half-precision floating points."""
  def __init__(self):
    super(TransformerFP16, self).__init__(dtype=tf.float16)

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
