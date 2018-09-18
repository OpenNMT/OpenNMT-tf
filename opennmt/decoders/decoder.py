"""Base class and functions for dynamic decoders."""

import abc
import six

import tensorflow as tf

from opennmt.layers.common import embedding_lookup
from opennmt.utils.beam_search import get_state_shape_invariants


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
             memory_sequence_length=None,
             return_alignment_history=False):
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
      return_alignment_history: If ``True``, also returns the alignment
        history from the attention layer (``None`` will be returned if
        unsupported by the decoder).

    Returns:
      A tuple ``(outputs, state, sequence_length)`` or
      ``(outputs, state, sequence_length, alignment_history)``
      if :obj:`return_alignment_history` is ``True``.
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


def greedy_decode(symbols_to_logits_fn,
                  initial_ids,
                  end_id,
                  decode_length=None,
                  state=None,
                  return_state=False):
  """Greedily decodes from :obj:`initial_ids`.

  Args:
    symbols_to_logits_fn: Interface to the model, to provide logits.
        Shoud take [batch_size, decoded_ids] and return [batch_size, vocab_size].
    initial_ids: Ids to start off the decoding, this will be the first thing
        handed to symbols_to_logits_fn.
    eos_id: ID for end of sentence.
    decode_length: Maximum number of steps to decode for.
    states: A dictionnary of (possibly nested) decoding states.
    return_state: If ``True``, also return the updated decoding states.

  Returns:
    A tuple with the decoded output, the decoded lengths, the log probabilities,
    and the decoding states (if :obj:`return_state` is ``True``).
  """
  batch_size = tf.shape(initial_ids)[0]
  finished = tf.tile([False], [batch_size])
  step = tf.constant(0)
  inputs = tf.expand_dims(initial_ids, 1)
  lengths = tf.zeros([batch_size], dtype=tf.int32)
  log_probs = tf.zeros([batch_size])

  def _condition(unused_step, finished, unused_inputs,
                 unused_lengths, unused_log_probs, unused_state):
    return tf.logical_not(tf.reduce_all(finished))

  def _body(step, finished, inputs, lengths, log_probs, state):
    inputs_lengths = tf.add(lengths, 1 - tf.cast(finished, lengths.dtype))

    logits, state = symbols_to_logits_fn(inputs, step, state)
    probs = tf.nn.log_softmax(logits)
    sample_ids = tf.argmax(probs, axis=-1)

    # Accumulate log probabilities.
    sample_probs = tf.reduce_max(probs, axis=-1)
    masked_probs = tf.squeeze(sample_probs, -1) * (1.0 - tf.cast(finished, sample_probs.dtype))
    log_probs = tf.add(log_probs, masked_probs)

    next_inputs = tf.concat([inputs, tf.cast(sample_ids, inputs.dtype)], -1)
    next_lengths = inputs_lengths
    next_finished = tf.logical_or(
        finished,
        tf.equal(tf.squeeze(sample_ids, axis=[1]), end_id))
    step = step + 1

    if decode_length is not None:
      next_finished = tf.logical_or(next_finished, step >= decode_length)

    return step, next_finished, next_inputs, next_lengths, log_probs, state

  _, _, outputs, lengths, log_probs, state = tf.while_loop(
      _condition,
      _body,
      loop_vars=(step, finished, inputs, lengths, log_probs, state),
      shape_invariants=(
          tf.TensorShape([]),
          finished.get_shape(),
          tf.TensorShape([None, None]),
          lengths.get_shape(),
          log_probs.get_shape(),
          tf.contrib.framework.nest.map_structure(get_state_shape_invariants, state)),
      parallel_iterations=1)

  if return_state:
    return outputs, lengths, log_probs, state
  return outputs, lengths, log_probs
