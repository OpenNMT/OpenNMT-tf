"""Base class and functions for dynamic decoders."""

import abc
import six

import tensorflow as tf

from opennmt.layers.common import embedding_lookup


def logits_to_cum_log_probs(logits, sequence_length):
  """Returns the cumulated log probabilities of sequences.

  Args:
    logits: The sequence of logits of shape :math:`[B, T, ...]`.
    sequence_length: The length of each sequence of shape :math:`[B]`.

  Returns:
    The cumulated log probability of each sequence.
  """
  mask = tf.sequence_mask(
      sequence_length, maxlen=tf.shape(logits)[1], dtype=logits.dtype)
  mask = tf.expand_dims(mask, -1)

  log_probs = tf.nn.log_softmax(logits)
  log_probs = log_probs * mask
  log_probs = tf.reduce_max(log_probs, axis=-1)
  log_probs = tf.reduce_sum(log_probs, axis=1)

  return log_probs

def get_embedding_fn(embedding):
  """Returns the embedding function.

  Args:
    embedding: The embedding tensor or a callable that takes word ids.

  Returns:
    A callable that takes word ids.
  """
  if callable(embedding):
    return embedding
  else:
    return lambda ids: embedding_lookup(embedding, ids)

def build_output_layer(num_units, vocab_size, dtype=None):
  """Builds the output projection layer.

  Args:
    num_units: The layer input depth.
    vocab_size: The layer output depth.
    dtype: The layer dtype.

  Returns:
    A ``tf.layers.Dense`` instance.

  Raises:
    ValueError: if :obj:`vocab_size` is ``None``.
  """
  if vocab_size is None:
    raise ValueError("vocab_size must be set to build the output layer")

  layer = tf.layers.Dense(vocab_size, use_bias=True, dtype=dtype)
  layer.build([None, num_units])
  return layer

def get_sampling_probability(global_step,
                             read_probability=None,
                             schedule_type=None,
                             k=None):
  """Returns the sampling probability as described in
  https://arxiv.org/abs/1506.03099.

  Args:
    global_step: The training step.
    read_probability: The probability to read from the inputs.
    schedule_type: The type of schedule.
    k: The convergence constant.

  Returns:
    The probability to sample from the output ids as a 0D ``tf.Tensor`` or
    ``None`` if scheduled sampling is not configured.

  Raises:
    ValueError: if :obj:`schedule_type` is set but not :obj:`k` or if
     :obj:`schedule_type` is ``linear`` but an initial :obj:`read_probability`
     is not set.
    TypeError: if :obj:`schedule_type` is invalid.
  """
  if read_probability is None and schedule_type is None:
    return None

  if schedule_type is not None and schedule_type != "constant":
    if k is None:
      raise ValueError("scheduled_sampling_k is required when scheduled_sampling_type is set")

    step = tf.cast(global_step, tf.float32)
    k = tf.constant(k, tf.float32)

    if schedule_type == "linear":
      if read_probability is None:
        raise ValueError("Linear schedule requires an initial read probability")
      read_probability = min(read_probability, 1.0)
      read_probability = tf.maximum(read_probability - k * step, 0.0)
    elif schedule_type == "exponential":
      read_probability = tf.pow(k, step)
    elif schedule_type == "inverse_sigmoid":
      read_probability = k / (k + tf.exp(step / k))
    else:
      raise TypeError("Unknown scheduled sampling type: {}".format(schedule_type))

  return 1.0 - read_probability


@six.add_metaclass(abc.ABCMeta)
class Decoder(object):
  """Base class for decoders."""

  @abc.abstractmethod
  def decode(self,
             inputs,
             sequence_length,
             vocab_size=None,
             initial_state=None,
             sampling_probability=None,
             embedding=None,
             output_layer=None,
             mode=tf.estimator.ModeKeys.TRAIN,
             memory=None,
             memory_sequence_length=None):
    """Decodes a full input sequence.

    Usually used for training and evaluation where target sequences are known.

    Args:
      inputs: The input to decode of shape :math:`[B, T, ...]`.
      sequence_length: The length of each input with shape :math:`[B]`.
      vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
        is not set.
      initial_state: The initial state as a (possibly nested tuple of...) tensors.
      sampling_probability: The probability of sampling categorically from
        the output ids instead of reading directly from the inputs.
      embedding: The embedding tensor or a callable that takes word ids.
        Must be set when :obj:`sampling_probability` is set.
      output_layer: Optional layer to apply to the output prior sampling.
        Must be set if :obj:`vocab_size` is not set.
      mode: A ``tf.estimator.ModeKeys`` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.

    Returns:
      A tuple ``(outputs, state, sequence_length)``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode(self,
                     embedding,
                     start_tokens,
                     end_token,
                     vocab_size=None,
                     initial_state=None,
                     output_layer=None,
                     maximum_iterations=250,
                     mode=tf.estimator.ModeKeys.PREDICT,
                     memory=None,
                     memory_sequence_length=None,
                     dtype=None,
                     return_alignment_history=False):
    """Decodes dynamically from :obj:`start_tokens` with greedy search.

    Usually used for inference.

    Args:
      embedding: The embedding tensor or a callable that takes word ids.
      start_tokens: The start token ids with shape :math:`[B]`.
      end_token: The end token id.
      vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
        is not set.
      initial_state: The initial state as a (possibly nested tuple of...) tensors.
      output_layer: Optional layer to apply to the output prior sampling.
        Must be set if :obj:`vocab_size` is not set.
      maximum_iterations: The maximum number of decoding iterations.
      mode: A ``tf.estimator.ModeKeys`` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.
      dtype: The data type. Required if :obj:`memory` is ``None``.
      return_alignment_history: If ``True``, also returns the alignment
        history from the attention layer (``None`` will be returned if
        unsupported by the decoder).

    Returns:
      A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
      ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def dynamic_decode_and_search(self,
                                embedding,
                                start_tokens,
                                end_token,
                                vocab_size=None,
                                initial_state=None,
                                output_layer=None,
                                beam_width=5,
                                length_penalty=0.0,
                                maximum_iterations=250,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=None,
                                memory_sequence_length=None,
                                dtype=None,
                                return_alignment_history=False):
    """Decodes dynamically from :obj:`start_tokens` with beam search.

    Usually used for inference.

    Args:
      embedding: The embedding tensor or a callable that takes word ids.
      start_tokens: The start token ids with shape :math:`[B]`.
      end_token: The end token id.
      vocab_size: The output vocabulary size. Must be set if :obj:`output_layer`
        is not set.
      initial_state: The initial state as a (possibly nested tuple of...) tensors.
      output_layer: Optional layer to apply to the output prior sampling.
        Must be set if :obj:`vocab_size` is not set.
      beam_width: The width of the beam.
      length_penalty: The length penalty weight during beam search.
      maximum_iterations: The maximum number of decoding iterations.
      mode: A ``tf.estimator.ModeKeys`` mode.
      memory: (optional) Memory values to query.
      memory_sequence_length: (optional) Memory values length.
      dtype: The data type. Required if :obj:`memory` is ``None``.
      return_alignment_history: If ``True``, also returns the alignment
        history from the attention layer (``None`` will be returned if
        unsupported by the decoder).

    Returns:
      A tuple ``(predicted_ids, state, sequence_length, log_probs)`` or
      ``(predicted_ids, state, sequence_length, log_probs, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
    """
    raise NotImplementedError()
