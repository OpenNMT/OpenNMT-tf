"""Language model."""

import tensorflow as tf

from opennmt import constants
from opennmt import inputters
from opennmt import layers
from opennmt.data import dataset as dataset_util
from opennmt.models.model import Model
from opennmt.utils import decoding
from opennmt.utils import losses
from opennmt.utils import misc


class LanguageModel(Model):
  """An experimental language model."""

  def __init__(self,
               decoder,
               embedding_size=None,
               reuse_embedding=True):
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
    super(LanguageModel, self).__init__(inputter)
    self.decoder = decoder
    self.reuse_embedding = reuse_embedding

  def auto_config(self, num_replicas=1):
    config = super(LanguageModel, self).auto_config(num_replicas=num_replicas)
    return misc.merge_dict(config, {
        "infer": {
            "length_bucket_width": 1  # To ensure fixed length in each batch.
        }
    })

  def build(self, input_shape):
    super(LanguageModel, self).build(input_shape)
    vocab_size = self.examples_inputter.vocabulary_size
    output_layer = None
    if self.reuse_embedding:
      output_layer = layers.Dense(
          vocab_size,
          weight=self.examples_inputter.embedding,
          transpose=True,
          dtype=self.examples_inputter.dtype)
    self.decoder.initialize(vocab_size=vocab_size, output_layer=output_layer)
    self.id_to_token = self.examples_inputter.vocabulary_lookup_reverse()

  def call(self, features, labels=None, training=None, step=None):
    outputs, predictions = None, None

    ids, length = features["ids"], features["length"]
    if labels is not None:
      # For training and evaluation, forward the full sequence.
      logits, _ = self._decode(ids, length, training=training)
      outputs = dict(logits=logits)
    else:
      assert_fixed_length = tf.debugging.Assert(
          tf.reduce_all(tf.equal(length, tf.reduce_max(length))),
          ["Language model does not support variable length contexts during "
           "generation, consider setting batch_size or length_bucket_width to 1"])

      # Run decoder one the context, if any.
      with tf.control_dependencies([assert_fixed_length]):
        context_ids, start_ids = tf.split(ids, [tf.shape(ids)[1] - 1, 1], axis=1)
        context_length = length - 1
        batch_size = tf.shape(context_length)[0]
        state = tf.cond(
            tf.equal(tf.reduce_sum(context_length), 0),
            true_fn=lambda: self.decoder.initial_state(batch_size=batch_size, dtype=self.dtype),
            false_fn=lambda: self._decode(context_ids, context_length)[1])

      # Iteratively decode from the last decoder state.
      sampled_ids, sampled_length, _, _, _ = decoding.dynamic_decode_from_params(
          self.decoder,
          self.examples_inputter,
          tf.squeeze(start_ids, 1),
          initial_state=state,
          params=self.params)
      sampled_ids = tf.reshape(sampled_ids, [batch_size, -1])
      sampled_length = tf.reshape(sampled_length, [batch_size])

      # Build the full prediction.
      full_ids = tf.concat([ids, sampled_ids], 1)
      full_length = length + sampled_length
      tokens = self.id_to_token.lookup(full_ids)
      predictions = dict(tokens=tokens, length=full_length)

    return outputs, predictions

  def _decode(self, ids, length_or_step, state=None, training=None):
    # Decode from ids.
    inputs = self.examples_inputter({"ids": ids}, training=training)
    logits, state, _ = self.decoder(inputs, length_or_step, state=state, training=training)
    return logits, state

  def compute_loss(self, outputs, labels, training=True):
    return losses.cross_entropy_sequence_loss(
        outputs["logits"],
        labels["ids_out"],
        labels["length"],
        label_smoothing=self.params.get("label_smoothing", 0.0),
        average_in_time=self.params.get("average_loss_in_time", False),
        training=training)

  def print_prediction(self, prediction, params=None, stream=None):
    target_length = prediction["length"]
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
        "ids_out": tf.concat([features["ids"][1:], [constants.END_OF_SENTENCE_ID]], 0),
        "length": tf.identity(features["length"])
    }
    if not training:
      labels["tokens"] = tf.concat([features["tokens"][1:], [constants.END_OF_SENTENCE_TOKEN]], 0)
    return features, labels

  def make_evaluation_dataset(self,
                              features_file,
                              labels_file,
                              batch_size,
                              num_threads=1,
                              prefetch_buffer_size=None):
    """See :meth:`opennmt.inputters.ExampleInputter.make_evaluation_dataset`."""
    _ = labels_file
    dataset = self.make_dataset(features_file, training=False)
    dataset = dataset.apply(dataset_util.inference_pipeline(
        batch_size,
        process_fn=lambda x: self._generate_example(x, training=False),
        num_threads=num_threads,
        prefetch_buffer_size=prefetch_buffer_size))
    return dataset

  def make_training_dataset(self,
                            features_file,
                            labels_file,
                            batch_size,
                            batch_type="examples",
                            batch_multiplier=1,
                            batch_size_multiple=1,
                            shuffle_buffer_size=None,
                            length_bucket_width=None,
                            maximum_features_length=None,
                            maximum_labels_length=None,
                            single_pass=False,
                            num_shards=1,
                            shard_index=0,
                            num_threads=4,
                            prefetch_buffer_size=None):
    """See :meth:`opennmt.inputters.ExampleInputter.make_training_dataset`."""
    _ = labels_file
    dataset = self.make_dataset(features_file, training=True)
    dataset = dataset.apply(dataset_util.training_pipeline(
        batch_size,
        batch_type=batch_type,
        batch_multiplier=batch_multiplier,
        length_bucket_width=length_bucket_width,
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
        shard_index=shard_index))
    return dataset
