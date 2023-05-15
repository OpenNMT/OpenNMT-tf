"""Catalog of predefined models."""

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import config as config_util
from opennmt import decoders, encoders, inputters, layers
from opennmt.models import (
    language_model,
    model,
    sequence_tagger,
    sequence_to_sequence,
    transformer,
)
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
        return config_util.merge_config(
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
        return config_util.merge_config(
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
        return config_util.merge_config(
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


@register_model_in_catalog(alias="Transformer")
class TransformerBase(transformer.Transformer):
    """Defines a base Transformer model as described in https://arxiv.org/abs/1706.03762."""


@register_model_in_catalog
class TransformerBaseSharedEmbeddings(transformer.Transformer):
    """Defines a base Transformer model with shared embeddings as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self):
        super().__init__(
            share_embeddings=sequence_to_sequence.EmbeddingsSharingLevel.ALL,
        )


@register_model_in_catalog(alias="TransformerRelative")
class TransformerBaseRelative(transformer.Transformer):
    """Defines a base Transformer model using relative position representations as
    described in https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(position_encoder_class=None, maximum_relative_position=20)


# Backward compatibility with model descriptions that directly accessed the catalog module.
Transformer = TransformerBase
TransformerRelative = TransformerBaseRelative


@register_model_in_catalog
class TransformerBig(transformer.Transformer):
    """Defines a big Transformer model as described in https://arxiv.org/abs/1706.03762."""

    def __init__(self):
        super().__init__(num_units=1024, num_heads=16, ffn_inner_dim=4096)


@register_model_in_catalog
class TransformerBigSharedEmbeddings(transformer.Transformer):
    """Defines a big Transformer model with shared embeddings as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(self):
        super().__init__(
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
            share_embeddings=sequence_to_sequence.EmbeddingsSharingLevel.ALL,
        )


@register_model_in_catalog
class TransformerBigRelative(transformer.Transformer):
    """Defines a big Transformer model using relative position representations as
    described in https://arxiv.org/abs/1803.02155.
    """

    def __init__(self):
        super().__init__(
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
            position_encoder_class=None,
            maximum_relative_position=20,
        )


@register_model_in_catalog
class TransformerTiny(transformer.Transformer):
    """Defines a tiny Transformer model."""

    def __init__(self):
        super().__init__(
            num_layers=2,
            num_units=64,
            num_heads=2,
            ffn_inner_dim=64,
        )


@register_model_in_catalog
class ScalingNmtEnDe(transformer.Transformer):
    """Defines a big Transformer model using the En-De hyperparameters from
    https://arxiv.org/abs/1806.00187.

    The architecture is equivalent to transformer_wmt_en_de_big in Fairseq.
    """

    def __init__(self, dropout=0.3, attention_dropout=0.1):
        super().__init__(
            num_layers=6,
            num_units=1024,
            num_heads=16,
            ffn_inner_dim=4096,
            pre_norm=False,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=0,
            share_embeddings=sequence_to_sequence.EmbeddingsSharingLevel.AUTO,
            output_layer_bias=False,
        )

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return config_util.merge_config(
            config,
            {
                "data": {
                    # Add EOS to the source.
                    "source_sequence_controls": {"end": True},
                },
                "params": {
                    "optimizer": "Adam",
                    "optimizer_params": {
                        "beta_1": 0.9,
                        "beta_2": 0.98,
                        "epsilon": 1e-8,
                    },
                    "learning_rate": 0.001,
                    "decay_type": "InvSqrtDecay",
                    "decay_params": {
                        "warmup_steps": 4000,
                        "initial_learning_rate": 1e-7,
                    },
                },
                "train": {
                    "batch_size": 0,
                    "effective_batch_size": 458752,  # = 3584 * 128
                    "maximum_features_length": 175,
                    "maximum_labels_length": 175,
                    "save_checkpoint_steps": 1000,
                    "keep_checkpoint_max": 10,
                    "average_last_checkpoints": 10,
                },
            },
        )


@register_model_in_catalog
class ScalingNmtEnFr(ScalingNmtEnDe):
    """Defines a big Transformer model using the En-Fr hyperparameters from
    https://arxiv.org/abs/1806.00187.

    The architecture is equivalent to transformer_vaswani_wmt_en_fr_big in Fairseq.
    """

    def __init__(self):
        super().__init__(dropout=0.1, attention_dropout=0)

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return config_util.merge_config(
            config,
            {
                "params": {"learning_rate": 0.0007},
                "train": {
                    "effective_batch_size": 655360,  # = 5120 * 128
                },
            },
        )


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
        return config_util.merge_config(
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
