"""Catalog of predefined models."""

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import decoders
from opennmt import encoders
from opennmt import inputters
from opennmt import layers
from opennmt.models import language_model
from opennmt.models import model
from opennmt.models import sequence_tagger
from opennmt.models import sequence_to_sequence
from opennmt.models import transformer
from opennmt.utils import misc


_CATALOG_MODELS_REGISTRY = misc.ClassRegistry(base_class=model.Model)

register_model_in_catalog = _CATALOG_MODELS_REGISTRY.register


def list_model_names_from_catalog():
    """Lists the models name registered in the catalog."""
    return _CATALOG_MODELS_REGISTRY.class_names


def get_model_from_catalog(name, as_builder=False):
    """Gets a model from the catalog.

    Args:
      name: The model name in the catalog.
      as_builder: If ``True``, return a callable building the model on call.

    Returns:
      A :class:`opennmt.models.Model` instance or a callable returning such
      instance.

    Raises:
      ValueError: if the model :obj:`name` does not exist in the catalog.
    """
    model_class = _CATALOG_MODELS_REGISTRY.get(name)
    if model_class is None:
        raise ValueError("The model '%s' does not exist in the model catalog" % name)
    if as_builder:
        return model_class
    return model_class()


@register_model_in_catalog
class ListenAttendSpell(sequence_to_sequence.SequenceToSequence):
    """Defines a model similar to the "Listen, Attend and Spell" model described
    in https://arxiv.org/abs/1508.01211.
    """

    def __init__(self):
        super().__init__(
            source_inputter=inputters.SequenceRecordInputter(input_depth=40),
            target_inputter=inputters.WordEmbedder(embedding_size=50),
            encoder=encoders.PyramidalRNNEncoder(
                num_layers=3,
                num_units=512,
                reduction_factor=2,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.3,
            ),
            decoder=decoders.AttentionalRNNDecoder(
                num_layers=3,
                num_units=512,
                attention_mechanism_class=tfa.seq2seq.LuongMonotonicAttention,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.3,
                residual_connections=False,
                first_layer_attention=True,
            ),
        )

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return misc.merge_dict(
            config,
            {
                "params": {
                    "optimizer": "SGD",
                    "learning_rate": 0.2,
                    "scheduled_sampling_type": "constant",
                    "scheduled_sampling_read_probability": 0.9,
                },
                "train": {
                    "batch_size": 32,
                    "length_bucket_width": 15,
                    "maximum_features_length": 2450,
                    "maximum_labels_length": 330,
                },
            },
        )


class _RNNBase(sequence_to_sequence.SequenceToSequence):
    """Base class for RNN based NMT models."""

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return misc.merge_dict(
            config,
            {
                "params": {
                    "optimizer": "Adam",
                    "learning_rate": 0.0002,
                },
                "train": {
                    "batch_size": 64,
                    "maximum_features_length": 80,
                    "maximum_labels_length": 80,
                },
            },
        )


@register_model_in_catalog
class LuongAttention(_RNNBase):
    """Defines a LSTM encoder-decoder model as described in https://arxiv.org/abs/1508.04025."""

    def __init__(self):
        super().__init__(
            source_inputter=inputters.WordEmbedder(embedding_size=512),
            target_inputter=inputters.WordEmbedder(embedding_size=512),
            encoder=encoders.RNNEncoder(
                num_layers=4,
                num_units=1000,
                dropout=0.2,
                residual_connections=False,
                cell_class=tf.keras.layers.LSTMCell,
            ),
            decoder=decoders.AttentionalRNNDecoder(
                num_layers=4,
                num_units=1000,
                bridge_class=layers.CopyBridge,
                attention_mechanism_class=tfa.seq2seq.LuongAttention,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.2,
                residual_connections=False,
            ),
        )


@register_model_in_catalog
class NMTBigV1(_RNNBase):
    """Defines a bidirectional LSTM encoder-decoder model.

    Note:
      For compatibility with OpenNMT-tf v1.
    """

    def __init__(self):
        super().__init__(
            source_inputter=inputters.WordEmbedder(embedding_size=512),
            target_inputter=inputters.WordEmbedder(embedding_size=512),
            encoder=encoders.RNNEncoder(
                num_layers=4,
                num_units=512,
                bidirectional=True,
                residual_connections=False,
                dropout=0.3,
                reducer=layers.ConcatReducer(),
                cell_class=tf.keras.layers.LSTMCell,
            ),
            decoder=decoders.AttentionalRNNDecoder(
                num_layers=4,
                num_units=1024,
                bridge_class=layers.CopyBridge,
                attention_mechanism_class=tfa.seq2seq.LuongAttention,
                attention_layer_activation=None,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.3,
                residual_connections=False,
            ),
        )


@register_model_in_catalog
class NMTMediumV1(_RNNBase):
    """Defines a medium-sized bidirectional LSTM encoder-decoder model.

    Note:
      For compatibility with OpenNMT-tf v1.
    """

    def __init__(self):
        super().__init__(
            source_inputter=inputters.WordEmbedder(embedding_size=512),
            target_inputter=inputters.WordEmbedder(embedding_size=512),
            encoder=encoders.RNNEncoder(
                num_layers=4,
                num_units=256,
                bidirectional=True,
                residual_connections=False,
                dropout=0.3,
                reducer=layers.ConcatReducer(),
                cell_class=tf.keras.layers.LSTMCell,
            ),
            decoder=decoders.AttentionalRNNDecoder(
                num_layers=4,
                num_units=512,
                bridge_class=layers.CopyBridge,
                attention_mechanism_class=tfa.seq2seq.LuongAttention,
                attention_layer_activation=None,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.3,
                residual_connections=False,
            ),
        )


@register_model_in_catalog
class NMTSmallV1(_RNNBase):
    """Defines a small unidirectional LSTM encoder-decoder model.

    Note:
      For compatibility with OpenNMT-tf v1.
    """

    def __init__(self):
        super().__init__(
            source_inputter=inputters.WordEmbedder(embedding_size=512),
            target_inputter=inputters.WordEmbedder(embedding_size=512),
            encoder=encoders.RNNEncoder(
                num_layers=2,
                num_units=512,
                residual_connections=False,
                dropout=0.3,
                cell_class=tf.keras.layers.LSTMCell,
            ),
            decoder=decoders.AttentionalRNNDecoder(
                num_layers=2,
                num_units=512,
                bridge_class=layers.CopyBridge,
                attention_mechanism_class=tfa.seq2seq.LuongAttention,
                attention_layer_activation=None,
                cell_class=tf.keras.layers.LSTMCell,
                dropout=0.3,
                residual_connections=False,
            ),
        )


@register_model_in_catalog
class LstmCnnCrfTagger(sequence_tagger.SequenceTagger):
    """Defines a bidirectional LSTM-CNNs-CRF as described in https://arxiv.org/abs/1603.01354."""

    def __init__(self):
        super().__init__(
            inputter=inputters.MixedInputter(
                [
                    inputters.WordEmbedder(embedding_size=100),
                    inputters.CharConvEmbedder(
                        embedding_size=30,
                        num_outputs=30,
                        kernel_size=3,
                        stride=1,
                        dropout=0.5,
                    ),
                ],
                dropout=0.5,
            ),
            encoder=encoders.RNNEncoder(
                num_layers=1,
                num_units=400,
                bidirectional=True,
                dropout=0.5,
                residual_connections=False,
                cell_class=tf.keras.layers.LSTMCell,
            ),
            crf_decoding=True,
        )

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return misc.merge_dict(
            config,
            {
                "params": {
                    "optimizer": "Adam",
                    "learning_rate": 0.001,
                },
                "train": {
                    "batch_size": 32,
                },
            },
        )


class _DefaultTransformer(transformer.Transformer):
    def __init__(self, big=False, relative=False, shared_embeddings=False):
        if big:
            num_units = 1024
            num_heads = 16
            ffn_inner_dim = 4096
        else:
            num_units = 512
            num_heads = 8
            ffn_inner_dim = 2048
        if relative:
            position_encoder_class = None
            maximum_relative_position = 20
        else:
            position_encoder_class = layers.SinusoidalPositionEncoder
            maximum_relative_position = None
        super().__init__(
            source_inputter=inputters.WordEmbedder(embedding_size=num_units),
            target_inputter=inputters.WordEmbedder(embedding_size=num_units),
            num_layers=6,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            position_encoder_class=position_encoder_class,
            share_embeddings=(
                sequence_to_sequence.EmbeddingsSharingLevel.ALL
                if shared_embeddings
                else sequence_to_sequence.EmbeddingsSharingLevel.NONE
            ),
            maximum_relative_position=maximum_relative_position,
        )


@register_model_in_catalog(alias="Transformer")
class TransformerBase(_DefaultTransformer):
    """Defines a base Transformer model as described in https://arxiv.org/abs/1706.03762."""


@register_model_in_catalog()
class TransformerBaseSharedEmbeddings(_DefaultTransformer):
    """Defines a base Transformer model with shared embeddings as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self):
        super().__init__(shared_embeddings=True)


@register_model_in_catalog(alias="TransformerRelative")
class TransformerBaseRelative(_DefaultTransformer):
    """Defines a base Transformer model using relative position representations as
    described in https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(relative=True)


# Backward compatibility with model descriptions that directly accessed the catalog module.
Transformer = TransformerBase
TransformerRelative = TransformerBaseRelative


@register_model_in_catalog
class TransformerBig(_DefaultTransformer):
    """Defines a big Transformer model as described in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(big=True)


@register_model_in_catalog
class TransformerBigSharedEmbeddings(_DefaultTransformer):
    """Defines a big Transformer model with shared embeddings as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self):
        super().__init__(big=True, shared_embeddings=True)


@register_model_in_catalog
class TransformerBigRelative(_DefaultTransformer):
    """Defines a big Transformer model using relative position representations as
    described in https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(big=True, relative=True)


@register_model_in_catalog
class GPT2Small(language_model.LanguageModel):
    """GPT-2 language model (small version) as described in:

    https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf
    """

    def __init__(self):
        super().__init__(
            decoder=decoders.SelfAttentionDecoder(
                num_layers=12,
                num_units=768,
                num_heads=12,
                ffn_inner_dim=3072,
                ffn_activation=layers.gelu,
                position_encoder_class=lambda: layers.PositionEmbedder(
                    maximum_position=1024
                ),
                num_sources=0,
            ),
            embedding_size=768,
        )

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return misc.merge_dict(
            config,
            {
                "params": {
                    "average_loss_in_time": True,
                    "optimizer": "Adam",
                    "learning_rate": 2.5e-4,
                    "decay_type": "CosineAnnealing",
                    "decay_params": {
                        "max_step": 1000000,
                        "warmup_steps": 2000,
                    },
                },
                "train": {
                    # Below options are from GPT-1.
                    "batch_size": 64,
                    "maximum_features_length": 512,
                },
            },
        )
