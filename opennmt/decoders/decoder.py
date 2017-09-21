import abc
import six

import tensorflow as tf


def logits_to_cum_log_probs(logits, sequence_length):
  """Returns the cumulated log probabilities of sequences.

  Args:
    logits: The sequence of logits of shape [B, T, ...].
    sequence_length: The length of each sequence of shape [B].

  Returns:
    The cumulated log probability of each sequence.
  """
  mask = tf.sequence_mask(sequence_length, dtype=tf.float32)
  mask = tf.expand_dims(mask, -1)

  log_probs = tf.nn.log_softmax(logits)
  log_probs = log_probs * mask
  log_probs = tf.reduce_max(log_probs, axis=-1)
  log_probs = tf.reduce_sum(log_probs, axis=1)

  return log_probs

def get_embedding_fn(embedding):
  """Returns the embedding function.

  Args:
    embeddings: The embeddings tensor or a callable that takes word ids.

  Returns:
    A callable that takes word ids.
  """
  if callable(embedding):
    return embedding
  else:
    return lambda ids: tf.nn.embedding_lookup(embedding, ids)


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
  """Base class for decoders."""

  @abc.abstractmethod
  def decode(self,
             inputs,
             sequence_length,
             vocab_size,
             encoder_state=None,
             scheduled_sampling_probability=0.0,
             embeddings=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    """Decodes a full input sequence.

    Usually used for training and evaluation where target sequences are known.

    Args:
      inputs: The input to decode of shape [B, T, ...].
      sequence_length: The length of each input with shape [B].
      vocab_size: The output vocabulary size.
      encoder_state: The encoder state as a (possibly nested tuple of...) tensors.
      scheduled_sampling_probability: The probability of sampling categorically from
        the output ids instead of reading directly from the inputs.
      embeddings: The embeddings tensor or a callable that takes word ids.
        Must be set when `scheduled_sampling_probability` > 0.
      mode: A `tf.estimator.ModeKeys` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.

    Returns:
      A tuple (`decoder_outputs`, `decoder_state`, `decoder_sequence_length`).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode(self,
                     embeddings,
                     start_tokens,
                     end_token,
                     vocab_size,
                     encoder_state=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.TRAIN,
                     memory=None,
                     memory_sequence_length=None):
    """Decodes dynamically from `start_tokens` with greedy search.

    Usually used for inference.

    Args:
      embeddings: The embeddings tensor or a callable that takes word ids.
      start_tokens: The start token ids with shape [B].
      end_token: The end token id.
      vocab_size: The output vocabulary size.
      encoder_state: The encoder state as a (possibly nested tuple of...) tensors.
      maximum_iterations: The maximum number of decoding iterations.
      mode: A `tf.estimator.ModeKeys` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.

    Returns:
      A tuple (`predicted_ids`, `state`, `sequence_length`, `log_probs`).
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode_and_search(self,
                                embeddings,
                                start_tokens,
                                end_token,
                                vocab_size,
                                encoder_state=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.TRAIN,
                                memory=None,
                                memory_sequence_length=None):
    """Decodes dynamically from `start_tokens` with beam search.

    Usually used for inference.

    Args:
      embeddings: The embeddings tensor or a callable that takes word ids.
      start_tokens: The start token ids with shape [B].
      end_token: The end token id.
      vocab_size: The output vocabulary size.
      encoder_state: The encoder state as a (possibly nested tuple of...) tensors.
      beam_width: The width of the beam.
      length_penalty: The length penalty weight during beam search.
      maximum_iterations: The maximum number of decoding iterations.
      mode: A `tf.estimator.ModeKeys` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.

    Returns:
      A tuple (`predicted_ids`, `state`, `sequence_length`, `log_probs`).
    """
    raise NotImplementedError()
