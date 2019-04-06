"""Language model."""

import tensorflow as tf

from opennmt import constants
from opennmt import inputters
from opennmt import layers
from opennmt.decoders import decoder as decoder_util
from opennmt.models.model import Model
from opennmt.utils import data
from opennmt.utils import decoding
from opennmt.utils import losses
from opennmt.utils import misc


class LanguageModel(Model):
  """An experimental language model."""

  def __init__(self,
               decoder,
               embedding_size=None,
               reuse_embedding=True,
               name="lm"):
    """Initializes the language model.

    Args:
      decoder: A :class:`opennmt.decoders.decoder.DecoderV2` instance.
      embedding_size: The size of the word embedding. If not set, pretrained
        embeddings should be defined in the configuration.
      reuse_embedding: If ``True``, reuse the embedding weights in the output
        layer.
      name: The name of this model.

    Raises:
      ValueError: if the decoder type is invalid.
    """
    if not isinstance(decoder, decoder_util.DecoderV2):
      raise ValueError("Language model only supports DecoderV2")
    inputter = LanguageModelInputter("vocabulary", embedding_size=embedding_size)
    super(LanguageModel, self).__init__(name, examples_inputter=inputter)
    self.decoder = decoder
    self.reuse_embedding = reuse_embedding

  def auto_config(self, num_devices=1):
    config = super(LanguageModel, self).auto_config(num_devices=num_devices)
    return misc.merge_dict(config, {
        "infer": {
            "bucket_width": 1  # To ensure fixed length in each batch.
        }
    })

  def _build(self):
    self.examples_inputter.build()
    vocab_size = self.examples_inputter.vocabulary_size
    output_layer = None
    if self.reuse_embedding:
      output_layer = layers.Dense(
          vocab_size,
          weight=self.examples_inputter.embedding,
          transpose=True,
          dtype=self.examples_inputter.dtype)
    self.decoder.initialize(vocab_size=vocab_size, output_layer=output_layer)

  def _call(self, features, labels, params, mode):
    training = mode == tf.estimator.ModeKeys.TRAIN
    outputs, predictions = None, None

    ids, length = features["ids"], features["length"]
    if mode != tf.estimator.ModeKeys.PREDICT:
      # For training and evaluation, forward the full sequence.
      logits, _ = self._decode(ids, length, training=training)
      outputs = dict(logits=logits)
    else:
      assert_fixed_length = tf.debugging.Assert(
          tf.reduce_all(tf.equal(length, tf.reduce_max(length))),
          ["Language model does not support variable length contexts during "
           "generation, consider setting batch_size or bucket_width to 1"])

      # Run decoder one the context, if any.
      with tf.control_dependencies([assert_fixed_length]):
        context_ids, start_ids = tf.split(ids, [tf.shape(ids)[1] - 1, 1], axis=1)
        context_length = length - 1
        batch_size = tf.shape(context_length)[0]
        state = tf.cond(
            tf.equal(tf.reduce_sum(context_length), 0),
            true_fn=lambda: self.decoder.get_initial_state(batch_size=batch_size, dtype=self.dtype),
            false_fn=lambda: self._decode(context_ids, context_length)[1],
            name=self.name + "/")  # Force the name scope.

      sampling_topk = params.get("sampling_topk")
      if sampling_topk is not None and sampling_topk != 1:
        sampler = decoding.RandomSampler(
            from_top_k=sampling_topk, temperature=params.get("sampling_temperature"))
      else:
        sampler = decoding.BestSampler()

      # Iteratively decode from the last decoder state.
      with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        sampled_ids, sampled_length, _, _, _ = decoding.dynamic_decode(
            self._decode,
            tf.squeeze(start_ids, 1),
            initial_state=state,
            sampler=sampler,
            maximum_iterations=params.get("maximum_iterations", 250),
            minimum_iterations=params.get("minimum_decoding_length", 0))
        sampled_ids = tf.squeeze(sampled_ids, 1)
        sampled_length = tf.squeeze(sampled_length, 1)

      # Build the full prediction.
      full_ids = tf.concat([ids, sampled_ids], 1)
      full_length = length + sampled_length
      vocab_rev = self.examples_inputter.vocabulary_lookup_reverse()
      tokens = vocab_rev.lookup(full_ids)
      predictions = dict(tokens=tokens, length=full_length)

    return outputs, predictions

  def _decode(self, ids, length_or_step, state=None, training=None):
    # Decode from ids.
    inputs = self.examples_inputter.make_inputs({"ids": ids}, training=training)
    logits, state, _ = self.decoder(inputs, length_or_step, state=state, training=training)
    return logits, state

  def compute_loss(self, outputs, labels, training=True, params=None):
    if params is None:
      params = {}
    return losses.cross_entropy_sequence_loss(
        outputs["logits"],
        labels["ids_out"],
        labels["length"],
        label_smoothing=params.get("label_smoothing", 0.0),
        average_in_time=params.get("average_loss_in_time", False),
        training=training)

  def print_prediction(self, prediction, params=None, stream=None):
    target_length = prediction["length"] - 1  # Ignore </s>.
    tokens = prediction["tokens"][:target_length]
    sentence = self.examples_inputter.tokenizer.detokenize(tokens)
    sentence = misc.format_translation_output(sentence)
    misc.print_bytes(tf.compat.as_bytes(sentence), stream=stream)


class LanguageModelInputter(inputters.WordEmbedder):
  """A special inputter for language modeling.

  This is a single word embedder that simply produces labels by shifting the
  input sequence.
  """

  def _generate_example(self, element, training=None):
    features = self.make_features(element, training=training)
    labels = {
        "tokens": tf.concat([features["tokens"][1:], [constants.END_OF_SENTENCE_TOKEN]], 0),
        "ids_out": tf.concat([features["ids"][1:], [constants.END_OF_SENTENCE_ID]], 0),
        "length": tf.identity(features["length"])
    }
    return features, labels

  def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              num_threads=1,
                              prefetch_buffer_size=None):
    """See :meth:`opennmt.inputters.inputter.ExampleInputter.make_evaluation_dataset`."""
    _ = labels_file
    dataset = self.make_dataset(features_file, training=False)
    dataset = data.inference_pipeline(
        dataset,
        batch_size,
        process_fn=lambda x: self._generate_example(x, training=False),
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size)
    return dataset

  def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            batch_type="examples",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
    """See :meth:`opennmt.inputters.inputter.ExampleInputter.make_training_dataset`."""
    _ = labels_file
    dataset = self.make_dataset(features_file, training=True)
    dataset = data.training_pipeline(
        dataset,
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        bucket_width=bucket_width,
        single_pass=single_pass,
        process_fn=lambda x: self._generate_example(x, training=True),
        num_threads=num_threads,
        shuffle_buffer_size=shuffle_buffer_size,
        prefetch_buffer_size=prefetch_buffer_size,
        maximum_features_length=maximum_features_length,
        maximum_labels_length=maximum_labels_length,
        features_length_fn=self.get_length,
        batch_size_multiple=batch_size_multiple,
        num_shards=num_shards,
        shard_index=shard_index)
    return dataset
