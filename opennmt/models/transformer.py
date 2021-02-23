"""Define the Google's Transformer model."""

import tensorflow as tf

from opennmt.models.sequence_to_sequence import (
    SequenceToSequence,
    EmbeddingsSharingLevel,
)
from opennmt.encoders.encoder import ParallelEncoder
from opennmt.encoders.self_attention_encoder import SelfAttentionEncoder
from opennmt.decoders.self_attention_decoder import SelfAttentionDecoder
from opennmt.layers.position import SinusoidalPositionEncoder
from opennmt.layers.transformer import MultiHeadAttentionReduction
from opennmt.utils.misc import merge_dict


class Transformer(SequenceToSequence):
    """Attention-based sequence-to-sequence model as described in
    https://arxiv.org/abs/1706.03762.
    """

    def __init__(
        self,
        source_inputter,
        target_inputter,
        num_layers,
        num_units,
        num_heads,
        ffn_inner_dim,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        ffn_activation=tf.nn.relu,
        position_encoder_class=SinusoidalPositionEncoder,
        share_embeddings=EmbeddingsSharingLevel.NONE,
        share_encoders=False,
        maximum_relative_position=None,
        attention_reduction=MultiHeadAttentionReduction.FIRST_HEAD_LAST_LAYER,
        pre_norm=True,
    ):
        """Initializes a Transformer model.

        Args:
          source_inputter: A :class:`opennmt.inputters.Inputter` to process
            the source data. If this inputter returns parallel inputs, a multi
            source Transformer architecture will be constructed.
          target_inputter: A :class:`opennmt.inputters.Inputter` to process
            the target data. Currently, only the
            :class:`opennmt.inputters.WordEmbedder` is supported.
          num_layers: The number of layers or a 2-tuple with the number of encoder
            layers and decoder layers.
          num_units: The number of hidden units.
          num_heads: The number of heads in each self-attention layers.
          ffn_inner_dim: The inner dimension of the feed forward layers.
          dropout: The probability to drop units in each layer output.
          attention_dropout: The probability to drop units from the attention.
          ffn_dropout: The probability to drop units from the ReLU activation in
            the feed forward layer.
          ffn_activation: The activation function to apply between the two linear
            transformations of the feed forward layer.
          position_encoder_class: The :class:`opennmt.layers.PositionEncoder`
            class to use for position encoding (or a callable that returns an
            instance).
          share_embeddings: Level of embeddings sharing, see
            :class:`opennmt.models.EmbeddingsSharingLevel` for possible values.
          share_encoders: In case of multi source architecture, whether to share the
            separate encoders parameters or not.
          maximum_relative_position: Maximum relative position representation
            (from https://arxiv.org/abs/1803.02155).
          attention_reduction: A :class:`opennmt.layers.MultiHeadAttentionReduction`
            value to specify how to reduce target-source multi-head attention
            matrices.
          pre_norm: If ``True``, layer normalization is applied before each
            sub-layer. Otherwise it is applied after. The original paper uses
            ``pre_norm=False``, but the authors later suggested that ``pre_norm=True``
            "seems better for harder-to-learn models, so it should probably be the
            default."
        """
        if isinstance(num_layers, (list, tuple)):
            num_encoder_layers, num_decoder_layers = num_layers
        else:
            num_encoder_layers, num_decoder_layers = num_layers, num_layers
        encoders = [
            SelfAttentionEncoder(
                num_encoder_layers,
                num_units=num_units,
                num_heads=num_heads,
                ffn_inner_dim=ffn_inner_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                position_encoder_class=position_encoder_class,
                maximum_relative_position=maximum_relative_position,
                pre_norm=pre_norm,
            )
            for _ in range(source_inputter.num_outputs)
        ]
        if len(encoders) > 1:
            encoder = ParallelEncoder(
                encoders if not share_encoders else encoders[0],
                outputs_reducer=None,
                states_reducer=None,
            )
        else:
            encoder = encoders[0]
        decoder = SelfAttentionDecoder(
            num_decoder_layers,
            num_units=num_units,
            num_heads=num_heads,
            ffn_inner_dim=ffn_inner_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            ffn_dropout=ffn_dropout,
            ffn_activation=ffn_activation,
            position_encoder_class=position_encoder_class,
            num_sources=source_inputter.num_outputs,
            maximum_relative_position=maximum_relative_position,
            attention_reduction=attention_reduction,
            pre_norm=pre_norm,
        )

        self._num_units = num_units
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._num_heads = num_heads
        self._with_relative_position = maximum_relative_position is not None
        self._is_ct2_compatible = (
            isinstance(encoder, SelfAttentionEncoder)
            and pre_norm
            and ffn_activation is tf.nn.relu
            and (
                (self._with_relative_position and position_encoder_class is None)
                or (
                    not self._with_relative_position
                    and position_encoder_class == SinusoidalPositionEncoder
                )
            )
        )
        super().__init__(
            source_inputter,
            target_inputter,
            encoder,
            decoder,
            share_embeddings=share_embeddings,
        )

    @property
    def ctranslate2_spec(self):
        if not self._is_ct2_compatible:
            return None
        import ctranslate2

        model_spec = ctranslate2.specs.TransformerSpec(
            (self._num_encoder_layers, self._num_decoder_layers),
            self._num_heads,
            with_relative_position=self._with_relative_position,
        )
        model_spec.with_source_bos = bool(self.features_inputter.mark_start)
        model_spec.with_source_eos = bool(self.features_inputter.mark_end)
        return model_spec

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return merge_dict(
            config,
            {
                "params": {
                    "average_loss_in_time": True,
                    "label_smoothing": 0.1,
                    "optimizer": "LazyAdam",
                    "optimizer_params": {"beta_1": 0.9, "beta_2": 0.998},
                    "learning_rate": 2.0,
                    "decay_type": "NoamDecay",
                    "decay_params": {
                        "model_dim": self._num_units,
                        "warmup_steps": 8000,
                    },
                },
                "train": {
                    "effective_batch_size": 25000,
                    "batch_size": 3072,
                    "batch_type": "tokens",
                    "maximum_features_length": 100,
                    "maximum_labels_length": 100,
                    "keep_checkpoint_max": 8,
                    "average_last_checkpoints": 8,
                },
            },
        )

    def map_v1_weights(self, weights):
        weights["seq2seq"] = weights.pop("transformer")
        return super().map_v1_weights(weights)
