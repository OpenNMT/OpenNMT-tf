"""Dynamic decoding utilities."""

import abc
import six

import tensorflow as tf

from opennmt import constants
from opennmt.utils import compat
from opennmt.utils import misc


@six.add_metaclass(abc.ABCMeta)
class Sampler(object):
  """Base class for samplers."""

  @abc.abstractmethod
  def __call__(self, scores, num_samples=1):
    """Samples predictions.

    Args:
      scores: The scores to sample from, a tensor of shape
        ``[batch_size, vocab_size]``.
      num_samples: The number of samples per batch to produce.

    Returns:
      sample_ids: The sampled ids.
      sample_scores: The sampled scores.
    """
    raise NotImplementedError()

class RandomSampler(Sampler):
  """Randomly samples from model outputs."""

  def __init__(self, from_top_k=None, temperature=None):
    """Initializes the random sampler.

    Args:
      from_top_k: Sample from the top K predictions instead of the full
        distribution.
      temperature: Divide logits by this value. High temperatures generate more
        random samples.
    """
    if from_top_k is not None and from_top_k <= 0:
      from_top_k = None
    self.from_top_k = from_top_k
    self.temperature = temperature

  def __call__(self, scores, num_samples=1):
    if self.from_top_k is None:
      sample_ids = _sample_from(scores, num_samples, temperature=self.temperature)
    else:
      top_scores, top_ids = tf.nn.top_k(scores, k=self.from_top_k)
      sample_ids = _sample_from(top_scores, num_samples, temperature=self.temperature)
      sample_ids = _gather_from_word_indices(top_ids, sample_ids)
    sample_scores = _gather_from_word_indices(scores, sample_ids)
    return sample_ids, sample_scores

class BestSampler(Sampler):
  """Sample the best predictions."""

  def __call__(self, scores, num_samples=1):
    sample_scores, sample_ids = tf.nn.top_k(scores, k=num_samples)
    return sample_ids, sample_scores


@six.add_metaclass(abc.ABCMeta)
class DecodingStrategy(object):
  """Base class for decoding strategies."""

  @property
  def num_hypotheses(self):
    """The number of hypotheses returned by this strategy."""
    return 1

  @abc.abstractmethod
  def initialize(self, batch_size, start_ids):
    """Initializes the strategy.

    Args:
      batch_size: The batch size.
      start_ids: The start decoding ids.

    Returns:
      start_ids: The (possibly transformed) start decoding ids.
      finished: The tensor of finished flags.
      initial_log_probs: Initial log probabilities per batch.
      extra_vars: A sequence of additional tensors used during the decoding.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def step(self, step, sampler, log_probs, cum_log_probs, finished, state, extra_vars):
    """Updates the strategy state.

    Args:
      step: The current decoding step.
      sampler: The sampler that produces predictions.
      log_probs: The model log probabilities.
      cum_log_probs: The cumulated log probabilities per batch.
      finished: The current finished flags.
      state: The decoder state.
      extra_vars: Additional tensors from this decoding strategy.

    Returns:
      ids: The predicted word ids.
      cum_log_probs: The new cumulated log probabilities.
      finished: The updated finished flags.
      state: The update decoder state.
      extra_vars: Additional tensors from this decoding strategy.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def finalize(self, outputs, end_id, extra_vars, attention=None):
    """Finalize the predictions.

    Args:
      outputs: The array of sampled ids.
      end_id: The end token id.
      extra_vars: Additional tensors from this decoding strategy.
      attention: The array of attention outputs.

    Returns:
      final_ids: The final predictions as a tensor of shape [B, H, T].
      final_attention: The final attention history of shape [B, H, T, S].
      final_lengths: The final sequence lengths of shape [B, H].
    """
    raise NotImplementedError()


class GreedySearch(DecodingStrategy):
  """A basic greedy search strategy."""

  def initialize(self, batch_size, start_ids):
    finished = tf.zeros([batch_size], dtype=tf.bool)
    initial_log_probs = tf.zeros([batch_size], dtype=tf.float32)
    return start_ids, finished, initial_log_probs, []

  def step(self, step, sampler, log_probs, cum_log_probs, finished, state, extra_vars):
    sample_ids, sample_log_probs = sampler(log_probs)
    sample_ids = tf.reshape(sample_ids, [-1])
    sample_log_probs = tf.reshape(sample_log_probs, [-1])
    cum_log_probs += sample_log_probs
    return sample_ids, cum_log_probs, finished, state, extra_vars

  def finalize(self, outputs, end_id, extra_vars, attention=None):
    ids = tf.transpose(outputs.stack())
    ids = tf.expand_dims(ids, 1)
    lengths = _lengths_from_ids(ids, end_id)
    if attention is not None:
      attention = tf.transpose(attention.stack(), perm=[1, 0, 2])
      attention = tf.expand_dims(attention, 1)
    return ids, attention, lengths


class BeamSearch(DecodingStrategy):
  """A beam search strategy."""

  def __init__(self, beam_size, length_penalty=0):
    self.beam_size = beam_size
    self.length_penalty = length_penalty

  @property
  def num_hypotheses(self):
    return self.beam_size

  def initialize(self, batch_size, start_ids):
    start_ids = tf.contrib.seq2seq.tile_batch(start_ids, self.beam_size)
    finished = tf.zeros([batch_size * self.beam_size], dtype=tf.bool)
    # Give all probability to first beam for the first iteration.
    initial_log_probs = tf.tile([0.] + [-float("inf")] * (self.beam_size - 1), [batch_size])
    parent_ids = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
    maximum_lengths = tf.zeros([batch_size], dtype=tf.int32)
    return start_ids, finished, initial_log_probs, (parent_ids, maximum_lengths)

  def step(self, step, sampler, log_probs, cum_log_probs, finished, state, extra_vars):
    parent_ids, maximum_lengths = extra_vars

    # Compute scores from log probabilities.
    vocab_size = log_probs.shape.as_list()[-1]
    total_probs = log_probs + tf.expand_dims(cum_log_probs, 1)  # Add current beam probability.
    total_probs = tf.reshape(total_probs, [-1, self.beam_size * vocab_size])
    scores = total_probs
    if self.length_penalty != 0:
      scores /= tf.pow(((5. + tf.cast(step + 1, tf.float32)) / 6.), self.length_penalty)

    # Sample predictions.
    sample_ids, sample_scores = sampler(scores, num_samples=self.beam_size)
    sample_ids = tf.reshape(sample_ids, [-1])
    sample_scores = tf.reshape(sample_scores, [-1])

    # Resolve beam origin and word ids.
    word_ids = sample_ids % vocab_size
    beam_ids = sample_ids // vocab_size
    beam_indices = (
        (tf.range(tf.shape(word_ids)[0]) // self.beam_size)
        * self.beam_size + beam_ids)

    # Update maximum length of unfinished batches.
    finished_batch = tf.reduce_all(tf.reshape(finished, [-1, self.beam_size]), axis=-1)
    maximum_lengths = tf.where(finished_batch, x=maximum_lengths, y=maximum_lengths + 1)

    # Update state and flags.
    cum_log_probs = _gather_from_word_indices(total_probs, sample_ids)
    finished = tf.gather(finished, beam_indices)
    parent_ids = parent_ids.write(step, beam_ids)
    state = compat.nest.map_structure(lambda s: _gather_state(s, beam_indices), state)
    return word_ids, cum_log_probs, finished, state, (parent_ids, maximum_lengths)

  def finalize(self, outputs, end_id, extra_vars, attention=None):
    parent_ids, maximum_lengths = extra_vars
    max_time = outputs.size()
    array_shape = [max_time, -1, self.beam_size]
    step_ids = tf.reshape(outputs.stack(), array_shape)
    parent_ids = tf.reshape(parent_ids.stack(), array_shape)
    ids = tf.contrib.seq2seq.gather_tree(
        step_ids, parent_ids, maximum_lengths, end_id)
    ids = tf.transpose(ids, perm=[1, 2, 0])
    lengths = _lengths_from_ids(ids, end_id)
    if attention is not None:
      attention = _gather_tree_from_array(
          attention.stack(), parent_ids, lengths)
      attention = tf.transpose(attention, perm=[1, 0, 2])
      attention = tf.reshape(
          attention, [tf.shape(ids)[0], self.beam_size, max_time, -1])
    return ids, attention, lengths


def dynamic_decode(symbols_to_logits_fn,
                   start_ids,
                   end_id=constants.END_OF_SENTENCE_ID,
                   initial_state=None,
                   decoding_strategy=None,
                   sampler=None,
                   maximum_iterations=None,
                   minimum_iterations=0,
                   attention_history=False):
  """Dynamic decoding.

  Args:
    symbols_to_logits_fn: A callable taking ``(symbols, step, state)`` and
      returning ``(logits, state, attention)`` (``attention`` is optional).
    start_ids: Initial input IDs of shape :math:`[B]`.
    end_id: ID of the end of sequence token.
    initial_state: Initial decoder state.
    decoding_strategy: A :class:`opennmt.utils.decoding.DecodingStrategy`
      instance that define the decoding logic. Defaults to a greedy search.
    sampler: A :class:`opennmt.utils.decoding.Sampler` instance that samples
      predictions from the model output. Defaults to an argmax sampling.
    maximum_iterations: The maximum number of iterations to decode for.
    minimum_iterations: The minimum number of iterations to decode for.
    attention_history: Gather attention history during the decoding.

  Returns:
    ids: The predicted ids of shape :math:`[B, H, T]`.
    lengths: The produced sequences length of shape :math:`[B, H]`.
    log_probs: The cumulated log probabilities of shape :math:`[B, H]`.
    attention_history: The attention history of shape :math:`[B, H, T_t, T_s]`.
    state: The final decoding state.
  """
  if "maximum_iterations" not in misc.function_args(tf.while_loop):
    raise NotImplementedError("Unified decoding does not support TensorFlow 1.4. "
                              "Please update your TensorFlow installation or open "
                              "an issue for assistance.")
  if decoding_strategy is None:
    decoding_strategy = GreedySearch()
  if sampler is None:
    sampler = BestSampler()

  def _cond(step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars):  # pylint: disable=unused-argument
    return tf.reduce_any(tf.logical_not(finished))

  def _body(step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars):
    # Get log probs from the model.
    result = symbols_to_logits_fn(inputs, step, state)
    logits, state = result[0], result[1]
    attn = result[2] if len(result) > 2 else None
    logits = tf.cast(logits, tf.float32)

    # Penalize or force EOS.
    batch_size, vocab_size = misc.shape_list(logits)
    eos_max_prob = tf.one_hot(
        tf.fill([batch_size], end_id),
        vocab_size,
        on_value=logits.dtype.max,
        off_value=logits.dtype.min)
    logits = tf.cond(
        step < minimum_iterations,
        true_fn=lambda: _penalize_token(logits, end_id),
        false_fn=lambda: tf.where(
            tf.tile(tf.expand_dims(finished, 1), [1, vocab_size]),
            x=eos_max_prob,
            y=logits))
    log_probs = tf.nn.log_softmax(logits)

    # Run one decoding strategy step.
    output, next_cum_log_probs, finished, state, extra_vars = decoding_strategy.step(
        step,
        sampler,
        log_probs,
        cum_log_probs,
        finished,
        state,
        extra_vars)

    # Update loop vars.
    if attention_history:
      if attn is None:
        raise ValueError("attention_history is set but the model did not return attention")
      attention = attention.write(step, tf.cast(attn, tf.float32))
    outputs = outputs.write(step, output)
    cum_log_probs = tf.where(finished, x=cum_log_probs, y=next_cum_log_probs)
    finished = tf.logical_or(finished, tf.equal(output, end_id))
    return step + 1, finished, state, output, outputs, attention, cum_log_probs, extra_vars

  batch_size = tf.shape(start_ids)[0]
  ids_dtype = start_ids.dtype
  start_ids = tf.cast(start_ids, tf.int32)
  start_ids, finished, initial_log_probs, extra_vars = decoding_strategy.initialize(
      batch_size, start_ids)
  step = tf.constant(0, dtype=tf.int32)
  outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  attention = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

  _, _, state, _, outputs, attention, log_probs, extra_vars = tf.while_loop(
      _cond,
      _body,
      loop_vars=(
          step,
          finished,
          initial_state,
          start_ids,
          outputs,
          attention,
          initial_log_probs,
          extra_vars),
      shape_invariants=(
          step.shape,
          finished.shape,
          compat.nest.map_structure(_get_shape_invariants, initial_state),
          start_ids.shape,
          tf.TensorShape(None),
          tf.TensorShape(None),
          initial_log_probs.shape,
          compat.nest.map_structure(_get_shape_invariants, extra_vars)),
      parallel_iterations=1,
      back_prop=False,
      maximum_iterations=maximum_iterations)

  ids, attention, lengths = decoding_strategy.finalize(
      outputs,
      end_id,
      extra_vars,
      attention=attention if attention_history else None)
  if attention is not None:
    attention = attention[:, :, 1:]  # Ignore attention for <s>.
  log_probs = tf.reshape(log_probs, [batch_size, decoding_strategy.num_hypotheses])
  ids = tf.cast(ids, ids_dtype)
  return ids, lengths, log_probs, attention, state


def _gather_state(tensor, indices):
  """Gather batch indices from the tensor."""
  if isinstance(tensor, tf.TensorArray) or tensor.shape.ndims == 0:
    return tensor
  return tf.gather(tensor, indices)

def _get_shape_invariants(tensor):
  """Returns the shape of the tensor but sets middle dims to None."""
  if isinstance(tensor, tf.TensorArray):
    shape = None
  else:
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
      shape[i] = None
  return tf.TensorShape(shape)

def _penalize_token(log_probs, token_id, penalty=-1e7):
  """Penalize token probabilities."""
  depth = log_probs.get_shape().as_list()[-1]
  penalty = tf.one_hot([token_id], depth, on_value=tf.cast(penalty, log_probs.dtype))
  return log_probs + penalty

def _sample_from(logits, num_samples, temperature=None):
  """Sample N values from the unscaled probability distribution."""
  if temperature is not None:
    logits /= tf.cast(temperature, logits.dtype)
  distribution = tf.distributions.Categorical(logits=logits)
  samples = distribution.sample([num_samples])
  return tf.transpose(samples)

def _gather_from_word_indices(tensor, indices):
  """Index the depth dim of a 2D tensor."""
  output_shape = misc.shape_list(indices)
  batch_size = tf.shape(tensor)[0]
  num_indices = tf.size(indices) // batch_size
  batch_pos = tf.range(batch_size * num_indices) // num_indices
  tensor = tf.gather_nd(tensor, tf.stack([batch_pos, tf.reshape(indices, [-1])], axis=-1))
  tensor = tf.reshape(tensor, output_shape)
  return tensor

def _lengths_from_ids(ids, end_id):
  """Compute sequence lengths from word ids."""
  lengths = tf.not_equal(ids, end_id)
  lengths = tf.cast(lengths, tf.int32)
  lengths = tf.reduce_sum(lengths, axis=-1)
  return lengths

def _gather_tree_from_array(array, parent_ids, sequence_length):
  """Calculates the full beams for `TensorArray`s.

  Args:
    array: A stacked `TensorArray` of size `max_time` that contains `Tensor`s of
      shape `[batch_size, beam_width, s]` or `[batch_size * beam_width, s]`
      where `s` is the depth shape.
    parent_ids: The parent ids of shape `[max_time, batch_size, beam_width]`.
    sequence_length: The sequence length of shape `[batch_size, beam_width]`.

  Returns:
    A `Tensor` which is a stacked `TensorArray` of the same size and type as
    `array` and where beams are sorted in each `Tensor` according to
    `parent_ids`.
  """
  max_time = parent_ids.shape.dims[0].value or tf.shape(parent_ids)[0]
  batch_size = parent_ids.shape.dims[1].value or tf.shape(parent_ids)[1]
  beam_width = parent_ids.shape.dims[2].value or tf.shape(parent_ids)[2]

  # Generate beam ids that will be reordered by gather_tree.
  beam_ids = tf.expand_dims(tf.expand_dims(tf.range(beam_width), 0), 0)
  beam_ids = tf.tile(beam_ids, [max_time, batch_size, 1])

  max_sequence_lengths = tf.cast(tf.reduce_max(sequence_length, axis=1), tf.int32)
  sorted_beam_ids = tf.contrib.seq2seq.gather_tree(
      step_ids=beam_ids,
      parent_ids=parent_ids,
      max_sequence_lengths=max_sequence_lengths,
      end_token=beam_width + 1)

  # For out of range steps, simply copy the same beam.
  in_bound_steps = tf.transpose(
      tf.sequence_mask(sequence_length, maxlen=max_time),
      perm=[2, 0, 1])
  sorted_beam_ids = tf.where(in_bound_steps, x=sorted_beam_ids, y=beam_ids)

  # Generate indices for gather_nd.
  time_ind = tf.tile(
      tf.reshape(tf.range(max_time), [-1, 1, 1]),
      [1, batch_size, beam_width])
  batch_ind = tf.tile(
      tf.reshape(tf.range(batch_size), [-1, 1, 1]),
      [1, max_time, beam_width])
  batch_ind = tf.transpose(batch_ind, perm=[1, 0, 2])
  indices = tf.stack([time_ind, batch_ind, sorted_beam_ids], -1)

  # Gather from a tensor with collapsed additional dimensions.
  final_shape = tf.shape(array)
  array = tf.reshape(array, [max_time, batch_size, beam_width, -1])
  ordered = tf.gather_nd(array, indices)
  ordered = tf.reshape(ordered, final_shape)
  return ordered
