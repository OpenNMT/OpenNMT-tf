"""Language model."""

import tensorflow as tf

from opennmt import inputters
from opennmt.models import model
from opennmt.utils import decoding
from opennmt.utils import losses
from opennmt.utils import misc


class LanguageModel(model.SequenceGenerator):
    """A language model."""

    def __init__(self, decoder, embedding_size=None, reuse_embedding=True):
        """Initializes the language model.

        Args:
          decoder: A :class:`opennmt.decoders.Decoder` instance.
          embedding_size: The size of the word embedding. If not set, pretrained
            embeddings should be defined in the configuration.
          reuse_embedding: If ``True``, reuse the embedding weights in the output
            layer.

        Raises:
          ValueError: if the decoder type is invalid.
        """
        inputter = LanguageModelInputter(embedding_size=embedding_size)
        super().__init__(inputter)
        self.decoder = decoder
        self.reuse_embedding = reuse_embedding

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        return misc.merge_dict(
            config,
            {
                "infer": {
                    "length_bucket_width": 1  # To ensure fixed length in each batch.
                }
            },
        )

    def initialize(self, data_config, params=None):
        super().initialize(data_config, params=params)
        self.decoder.initialize(vocab_size=self.examples_inputter.vocabulary_size)

    def build(self, input_shape):
        super().build(input_shape)
        if self.reuse_embedding:
            self.decoder.reuse_embeddings(self.examples_inputter.embedding)

    def call(self, features, labels=None, training=None, step=None):
        outputs, predictions = None, None

        ids, length = features["ids"], features["length"]
        if labels is not None:
            # For training and evaluation, forward the full sequence.
            logits, _ = self._decode(
                labels.get("ids", ids), labels.get("length", length), training=training
            )
            outputs = dict(logits=logits)
        else:
            assert_fixed_length = tf.debugging.Assert(
                tf.reduce_all(tf.equal(length, tf.reduce_max(length))),
                [
                    "Language model does not support variable length contexts during "
                    "generation, consider setting batch_size or length_bucket_width to 1"
                ],
            )
            assert_non_empty_start = tf.debugging.Assert(
                tf.math.not_equal(tf.math.reduce_max(length), 0),
                [
                    "The language model requires a context sequence to initialize the decoding. "
                    "If you want nonconditional sequence generation, you should configure the "
                    "sequence_controls parameter before training."
                ],
            )

            # Run decoder on the context, if any.
            with tf.control_dependencies([assert_fixed_length, assert_non_empty_start]):
                context_ids, start_ids = tf.split(
                    ids, [tf.shape(ids)[1] - 1, 1], axis=1
                )
                context_length = length - 1
                batch_size = tf.shape(context_length)[0]
                state = tf.cond(
                    tf.equal(tf.reduce_sum(context_length), 0),
                    true_fn=lambda: self.decoder.initial_state(
                        batch_size=batch_size, dtype=self.dtype
                    ),
                    false_fn=lambda: self._decode(context_ids, context_length)[1],
                )

            params = self.params

            def _decode_with_step_offset(ids, step, state):
                return self._decode(ids, step + context_length[0], state)

            # Iteratively decode from the last decoder state.
            sampled_ids, sampled_length, _, _, _ = decoding.dynamic_decode(
                _decode_with_step_offset,
                tf.squeeze(start_ids, 1),
                initial_state=state,
                sampler=decoding.Sampler.from_params(params),
                maximum_iterations=params.get("maximum_decoding_length", 250),
                minimum_iterations=params.get("minimum_decoding_length", 0),
            )
            sampled_ids = tf.reshape(sampled_ids, [batch_size, -1])
            sampled_length = tf.reshape(sampled_length, [batch_size])

            # Build the full prediction.
            if self.features_inputter.mark_start:
                # Remove leading <s> if included in the context sequence.
                ids = ids[:, 1:]
                length -= 1
            full_ids = tf.concat([ids, sampled_ids], 1)
            full_length = length + sampled_length
            tokens = self.features_inputter.ids_to_tokens.lookup(full_ids)
            predictions = dict(tokens=tokens, length=full_length)

        return outputs, predictions

    def _decode(self, ids, length_or_step, state=None, training=None):
        # Decode from ids.
        inputs = self.examples_inputter({"ids": ids}, training=training)
        logits, state, _ = self.decoder(
            inputs, length_or_step, state=state, training=training
        )
        return logits, state

    def compute_loss(self, outputs, labels, training=True):
        return losses.cross_entropy_sequence_loss(
            outputs["logits"],
            labels["ids_out"],
            labels["length"],
            label_smoothing=self.params.get("label_smoothing", 0.0),
            average_in_time=self.params.get("average_loss_in_time", False),
            training=training,
        )

    def print_prediction(self, prediction, params=None, stream=None):
        target_length = prediction["length"]
        tokens = prediction["tokens"][:target_length]
        sentence = self.examples_inputter.tokenizer.detokenize(tokens)
        sentence = misc.format_translation_output(sentence)
        misc.print_as_bytes(sentence, stream=stream)


class LanguageModelInputter(inputters.WordEmbedder, inputters.ExampleInputterAdapter):
    """A special inputter for language modeling.

    This is a single word embedder that simply produces labels by shifting the
    input sequence.
    """

    def initialize(self, data_config):
        super().initialize(data_config)
        # Set default sequence controls for backward compatibility.
        if self.mark_start is None:
            self.mark_start = False
        if self.mark_end is None:
            self.mark_end = True

    def make_features(self, element=None, features=None, training=None):
        base_features = features if features is not None else {}

        # Features define the decoder context during inference. As the context is a prefix,
        # we should disable the end sequence control token.
        saved_mark_end = self.mark_end
        self.set_decoder_mode(enable=False, mark_end=False)
        features = super().make_features(
            element=element, features=base_features.copy(), training=training
        )

        # Labels define the decoder input/output sequences during training and evaluation.
        self.set_decoder_mode(enable=True, mark_end=saved_mark_end)
        labels = super().make_features(
            element=element, features=base_features.copy(), training=training
        )

        return features, labels
