"""Dynamic decoding utilities."""

import abc
import collections

import tensorflow as tf
import tensorflow_addons as tfa

from opennmt import constants
from opennmt.utils import misc


class Sampler(abc.ABC):
    """Base class for samplers."""

    @abc.abstractmethod
    def __call__(self, scores, num_samples=1):
        """Samples predictions.

        Args:
          scores: The scores to sample from, a tensor of shape
            ``[batch_size, vocab_size]``.
          num_samples: The number of samples per batch to produce.

        Returns:
          A tuple ``(sample_ids, sample_scores)``.
        """
        raise NotImplementedError()

    @staticmethod
    def from_params(params):
        """Constructs a sampler based on user parameters.

        Args:
          params: A dictionary of user parameters.

        Returns:
          A :class:`opennmt.utils.Sampler` instance.
        """
        sampling_topk = params.get("sampling_topk", 1)
        if sampling_topk == 1:
            return BestSampler()
        else:
            return RandomSampler(
                from_top_k=sampling_topk, temperature=params.get("sampling_temperature")
            )


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
            sample_ids = _sample_from(
                top_scores, num_samples, temperature=self.temperature
            )
            sample_ids = _gather_from_word_indices(top_ids, sample_ids)
        sample_scores = _gather_from_word_indices(scores, sample_ids)
        return sample_ids, sample_scores


class BestSampler(Sampler):
    """Sample the best predictions."""

    def __call__(self, scores, num_samples=1):
        sample_scores, sample_ids = tf.nn.top_k(scores, k=num_samples)
        return sample_ids, sample_scores


class DecodingStrategy(abc.ABC):
    """Base class for decoding strategies."""

    @property
    def num_hypotheses(self):
        """The number of hypotheses returned by this strategy."""
        return 1

    @staticmethod
    def from_params(params):
        """Constructs a decoding strategy based on user parameters.

        Args:
          params: A dictionary of user parameters.

        Returns:
          A :class:`opennmt.utils.DecodingStrategy` instance.
        """
        beam_size = params.get("beam_width", 1)
        if beam_size > 1:
            return BeamSearch(
                beam_size,
                length_penalty=params.get("length_penalty", 0),
                coverage_penalty=params.get("coverage_penalty", 0),
            )
        else:
            return GreedySearch()

    @abc.abstractmethod
    def initialize(self, start_ids, attention_size=None):
        """Initializes the strategy.

        Args:
          start_ids: The start decoding ids.
          attention_size: If known, the size of the attention vectors (i.e. the
            maximum source length).

        Returns:
          A tuple containing,

          - The (possibly transformed) start decoding ids.
          - The tensor of finished flags.
          - Initial log probabilities per batch.
          - An dictionary of additional tensors used during the decoding.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def step(
        self,
        step,
        sampler,
        log_probs,
        cum_log_probs,
        finished,
        state=None,
        attention=None,
        **kwargs
    ):
        """Updates the strategy state.

        Args:
          step: The current decoding step.
          sampler: The sampler that produces predictions.
          log_probs: The model log probabilities.
          cum_log_probs: The cumulated log probabilities per batch.
          finished: The current finished flags.
          state: The decoder state.
          attention: The attention vector for the current step.
          **kwargs: Additional tensors used by this decoding strategy.

        Returns:
          A tuple containing,

          - The predicted word ids.
          - The new cumulated log probabilities.
          - The updated finished flags.
          - The updated decoder state.
          - A dictionary with additional tensors used by this decoding strategy.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def finalize(self, outputs, end_id, attention=None, **kwargs):
        """Finalize the predictions.

        Args:
          outputs: The array of sampled ids.
          end_id: The end token id.
          attention: The array of attention outputs.
          **kwargs: Additional tensors used by this decoding strategy.

        Returns:
          A tuple containing,

          - The final predictions as a tensor of shape [B, H, T].
          - The final attention history of shape [B, H, T, S].
          - The final sequence lengths of shape [B, H].
        """
        raise NotImplementedError()


class GreedySearch(DecodingStrategy):
    """A basic greedy search strategy."""

    def initialize(self, start_ids, attention_size=None):
        batch_size = tf.shape(start_ids)[0]
        finished = tf.zeros([batch_size], dtype=tf.bool)
        initial_log_probs = tf.zeros([batch_size], dtype=tf.float32)
        return start_ids, finished, initial_log_probs, {}

    def step(
        self,
        step,
        sampler,
        log_probs,
        cum_log_probs,
        finished,
        state=None,
        attention=None,
        **kwargs
    ):
        sample_ids, sample_log_probs = sampler(log_probs)
        sample_ids = tf.reshape(sample_ids, [-1])
        sample_log_probs = tf.reshape(sample_log_probs, [-1])
        cum_log_probs += sample_log_probs
        return sample_ids, cum_log_probs, finished, state, kwargs

    def finalize(self, outputs, end_id, attention=None, **kwargs):
        ids = tf.transpose(outputs.stack())
        ids = tf.expand_dims(ids, 1)
        lengths = _lengths_from_ids(ids, end_id)
        if attention is not None:
            attention = tf.transpose(attention.stack(), perm=[1, 0, 2])
            attention = tf.expand_dims(attention, 1)
        return ids, attention, lengths


class BeamSearch(DecodingStrategy):
    """A beam search strategy."""

    def __init__(self, beam_size, length_penalty=0, coverage_penalty=0):
        """Initializes the decoding strategy.

        Args:
          beam_size: The number of paths to consider per batch.
          length_penalty: Length penalty, see https://arxiv.org/abs/1609.08144.
          coverage_penalty: Coverage penalty, see https://arxiv.org/abs/1609.08144.
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self._state_reorder_flags = None

    @property
    def num_hypotheses(self):
        return self.beam_size

    def _set_state_reorder_flags(self, state_reorder_flags):
        """Sets state reorder flags, a structure matching the decoder state that
        indicates which tensor should be reorded during beam search.
        """
        self._state_reorder_flags = state_reorder_flags

    def initialize(self, start_ids, attention_size=None):
        batch_size = tf.shape(start_ids)[0]
        start_ids = tfa.seq2seq.tile_batch(start_ids, self.beam_size)
        finished = tf.zeros([batch_size * self.beam_size], dtype=tf.bool)
        # Give all probability to first beam for the first iteration.
        initial_log_probs = tf.tile(
            [0.0] + [-float("inf")] * (self.beam_size - 1), [batch_size]
        )
        extra_vars = {
            "parent_ids": tf.TensorArray(tf.int32, size=0, dynamic_size=True),
            "sequence_lengths": tf.zeros([batch_size * self.beam_size], dtype=tf.int32),
        }
        if self.coverage_penalty != 0:
            if attention_size is None:
                raise ValueError(
                    "The attention size should be known to support coverage penalty"
                )
            extra_vars["accumulated_attention"] = tf.zeros(
                [batch_size * self.beam_size, attention_size]
            )
        return start_ids, finished, initial_log_probs, extra_vars

    def _get_scores(
        self, log_probs, sequence_lengths, finished, accumulated_attention=None
    ):
        scores = log_probs
        if self.length_penalty != 0:
            expand_sequence_lengths = tf.expand_dims(sequence_lengths, 1)
            scores /= tf.pow(
                ((5.0 + tf.cast(expand_sequence_lengths + 1, scores.dtype)) / 6.0),
                self.length_penalty,
            )
        if self.coverage_penalty != 0:
            # Mask out of range steps with ones (log(1) == 0).
            accumulated_attention = tf.where(
                tf.equal(accumulated_attention, 0.0),
                x=tf.ones_like(accumulated_attention),
                y=accumulated_attention,
            )
            coverage_penalty = tf.reduce_sum(
                tf.math.log(tf.minimum(accumulated_attention, 1.0)), 1
            )
            # Apply coverage penalty to finished predictions.
            coverage_penalty *= tf.cast(finished, coverage_penalty.dtype)
            scores += self.coverage_penalty * tf.expand_dims(coverage_penalty, 1)
        return scores

    def step(
        self,
        step,
        sampler,
        log_probs,
        cum_log_probs,
        finished,
        state=None,
        attention=None,
        **kwargs
    ):
        parent_ids = kwargs["parent_ids"]
        sequence_lengths = kwargs["sequence_lengths"]

        if self.coverage_penalty != 0:
            if attention is None:
                raise ValueError(
                    "Coverage penalty is enabled but the model did not "
                    "return an attention vector"
                )
            not_finished = tf.math.logical_not(finished)
            attention *= tf.expand_dims(tf.cast(not_finished, attention.dtype), 1)
            accumulated_attention = kwargs["accumulated_attention"] + attention
        else:
            accumulated_attention = None

        # Compute scores from log probabilities.
        vocab_size = log_probs.shape[-1]
        total_probs = log_probs + tf.expand_dims(
            cum_log_probs, 1
        )  # Add current beam probability.
        scores = self._get_scores(
            total_probs,
            sequence_lengths,
            finished,
            accumulated_attention=accumulated_attention,
        )
        scores = tf.reshape(scores, [-1, self.beam_size * vocab_size])
        total_probs = tf.reshape(total_probs, [-1, self.beam_size * vocab_size])

        # Sample predictions.
        sample_ids, sample_scores = sampler(scores, num_samples=self.beam_size)
        cum_log_probs = tf.reshape(
            _gather_from_word_indices(total_probs, sample_ids), [-1]
        )
        sample_ids = tf.reshape(sample_ids, [-1])
        sample_scores = tf.reshape(sample_scores, [-1])

        # Resolve beam origin and word ids.
        word_ids = sample_ids % vocab_size
        beam_ids = sample_ids // vocab_size
        beam_indices = (
            tf.range(tf.shape(word_ids)[0]) // self.beam_size
        ) * self.beam_size + beam_ids

        # Update sequence_length of unfinished sequence.
        sequence_lengths = tf.where(
            finished, x=sequence_lengths, y=sequence_lengths + 1
        )

        # Update state and flags.
        finished = tf.gather(finished, beam_indices)
        sequence_lengths = tf.gather(sequence_lengths, beam_indices)
        parent_ids = parent_ids.write(step, beam_ids)
        extra_vars = {
            "parent_ids": parent_ids,
            "sequence_lengths": sequence_lengths,
        }
        if accumulated_attention is not None:
            extra_vars["accumulated_attention"] = tf.gather(
                accumulated_attention, beam_indices
            )
        if state is not None:
            state = _reorder_state(
                state, beam_indices, reorder_flags=self._state_reorder_flags
            )
        return word_ids, cum_log_probs, finished, state, extra_vars

    # We prefer using the pure TensorFlow variant of gather_tree instead of the custom op,
    # which is not readily available in TensorFlow builds (e.g. TensorFlow Serving).
    @misc.disable_tfa_custom_ops
    def finalize(self, outputs, end_id, attention=None, **kwargs):
        parent_ids = kwargs["parent_ids"]
        sequence_lengths = kwargs["sequence_lengths"]
        maximum_lengths = tf.reduce_max(
            tf.reshape(sequence_lengths, [-1, self.beam_size]), axis=-1
        )
        max_time = outputs.size()
        array_shape = [max_time, -1, self.beam_size]
        step_ids = tf.reshape(outputs.stack(), array_shape)
        parent_ids = tf.reshape(parent_ids.stack(), array_shape)
        ids = tfa.seq2seq.gather_tree(step_ids, parent_ids, maximum_lengths, end_id)
        ids = tf.transpose(ids, perm=[1, 2, 0])
        lengths = _lengths_from_ids(ids, end_id)
        if attention is not None:
            attention = tfa.seq2seq.gather_tree_from_array(
                attention.stack(), parent_ids, lengths
            )
            attention = tf.transpose(attention, perm=[1, 0, 2])
            attention = tf.reshape(
                attention, [tf.shape(ids)[0], self.beam_size, max_time, -1]
            )
        return ids, attention, lengths


class DecodingResult(
    collections.namedtuple(
        "DecodingResult", ("ids", "lengths", "log_probs", "attention", "state")
    )
):
    """Final decoding result.

    Args:
      ids: The predicted ids of shape :math:`[B, H, T]`.
      lengths: The produced sequences length of shape :math:`[B, H]`.
      log_probs: The cumulated log probabilities of shape :math:`[B, H]`.
      attention: The attention history of shape :math:`[B, H, T_t, T_s]`.
      state: The final decoding state.
    """


def dynamic_decode(
    symbols_to_logits_fn,
    start_ids,
    end_id=constants.END_OF_SENTENCE_ID,
    initial_state=None,
    decoding_strategy=None,
    sampler=None,
    maximum_iterations=None,
    minimum_iterations=0,
    attention_history=False,
    attention_size=None,
    tflite_output_size=None,
):
    """Dynamic decoding.

    Args:
      symbols_to_logits_fn: A callable taking ``(symbols, step, state)`` and
        returning ``(logits, state, attention)`` (``attention`` is optional).
      start_ids: Initial input IDs of shape :math:`[B]`.
      end_id: ID of the end of sequence token.
      initial_state: Initial decoder state.
      decoding_strategy: A :class:`opennmt.utils.DecodingStrategy`
        instance that defines the decoding logic. Defaults to a greedy search.
      sampler: A :class:`opennmt.utils.Sampler` instance that samples
        predictions from the model output. Defaults to an argmax sampling.
      maximum_iterations: The maximum number of iterations to decode for.
      minimum_iterations: The minimum number of iterations to decode for.
      attention_history: Gather attention history during the decoding.
      attention_size: If known, the size of the attention vectors (i.e. the
        maximum source length).
      tflite_output_size: If not None will run TFLite safe, is the size of 1D output tensor.

    Returns:
      A :class:`opennmt.utils.DecodingResult` instance.
    """
    if initial_state is None:
        initial_state = {}
    if decoding_strategy is None:
        decoding_strategy = GreedySearch()
    if sampler is None:
        sampler = BestSampler()
    is_tflite_run = tflite_output_size is not None

    def _cond(
        step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars
    ):
        return tf.reduce_any(tf.logical_not(finished))

    def _body(
        step, finished, state, inputs, outputs, attention, cum_log_probs, extra_vars
    ):
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
            off_value=logits.dtype.min,
        )
        logits = tf.cond(
            step < minimum_iterations,
            true_fn=lambda: _penalize_token(logits, end_id),
            false_fn=lambda: tf.where(
                tf.broadcast_to(tf.expand_dims(finished, -1), tf.shape(logits)),
                x=eos_max_prob,
                y=logits,
            ),
        )
        log_probs = tf.nn.log_softmax(logits)

        # Run one decoding strategy step.
        (
            output,
            next_cum_log_probs,
            finished,
            state,
            extra_vars,
        ) = decoding_strategy.step(
            step,
            sampler,
            log_probs,
            cum_log_probs,
            finished,
            state=state,
            attention=attn,
            **extra_vars,
        )

        # For TensorArray assuming batch size of 1 to be TFlite safe
        if is_tflite_run:
            outputs = outputs.write(step, output[0])
        else:
            # Update loop vars.
            if attention_history:
                if attn is None:
                    raise ValueError(
                        "attention_history is set but the model did not return attention"
                    )
                attention = attention.write(step, tf.cast(attn, tf.float32))
            outputs = outputs.write(step, output)
        cum_log_probs = tf.where(finished, x=cum_log_probs, y=next_cum_log_probs)
        finished = tf.logical_or(finished, tf.equal(output, end_id))
        return (
            step + 1,
            finished,
            state,
            output,
            outputs,
            attention,
            cum_log_probs,
            extra_vars,
        )

    start_ids = tf.convert_to_tensor(start_ids)
    ids_dtype = start_ids.dtype
    start_ids = tf.cast(start_ids, tf.int32)
    start_ids, finished, initial_log_probs, extra_vars = decoding_strategy.initialize(
        start_ids, attention_size=attention_size
    )
    step = tf.constant(0, dtype=tf.int32)

    if is_tflite_run:
        output_shape = tf.TensorShape(())
        outputs = tf.TensorArray(
            tf.int32,
            size=tflite_output_size,
            dynamic_size=False,
            element_shape=output_shape,
        )
        attn_shape = tf.TensorShape(())
        attention = tf.TensorArray(
            tf.float32,
            size=tflite_output_size,
            dynamic_size=False,
            element_shape=attn_shape,
        )
        maximum_iterations = (
            tflite_output_size
            if maximum_iterations > tflite_output_size
            else maximum_iterations
        )
    else:
        output_shape = tf.TensorShape(None)
        outputs = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
        attn_shape = tf.TensorShape(None)
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
            extra_vars,
        ),
        shape_invariants=(
            step.shape,
            finished.shape,
            tf.nest.map_structure(_get_shape_invariants, initial_state),
            start_ids.shape,
            output_shape,
            attn_shape,
            initial_log_probs.shape,
            tf.nest.map_structure(_get_shape_invariants, extra_vars),
        ),
        parallel_iterations=1,
        maximum_iterations=maximum_iterations,
    )

    ids, attention, lengths = decoding_strategy.finalize(
        outputs,
        end_id,
        attention=attention if attention_history and not is_tflite_run else None,
        **extra_vars,
    )
    if attention is not None and not is_tflite_run:
        attention = attention[:, :, :-1]  # Ignore attention for </s>.
    log_probs = tf.reshape(log_probs, [-1, decoding_strategy.num_hypotheses])
    ids = tf.cast(ids, ids_dtype)
    return DecodingResult(
        ids=ids, lengths=lengths, log_probs=log_probs, attention=attention, state=state
    )


def _reorder_state(state, indices, reorder_flags=None):
    """Gather batch indices from the state tensors."""

    def _reorder_one(tensor, reorder=True):
        if not reorder or isinstance(tensor, tf.TensorArray) or tensor.shape.ndims == 0:
            return tensor
        return tf.gather(tensor, indices)

    args = [state]
    if reorder_flags is not None:
        tf.nest.assert_same_structure(state, reorder_flags)
        args.append(reorder_flags)
    return tf.nest.map_structure(_reorder_one, *args)


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
    depth = log_probs.shape[-1]
    penalty = tf.one_hot([token_id], depth, on_value=tf.cast(penalty, log_probs.dtype))
    return log_probs + penalty


def _sample_from(logits, num_samples, temperature=None):
    """Sample N values from the unscaled probability distribution."""
    if temperature is not None:
        logits /= tf.cast(temperature, logits.dtype)
    return tf.random.categorical(logits, num_samples, dtype=tf.int32)


def _gather_from_word_indices(tensor, indices):
    """Index the depth dim of a 2D tensor."""
    return tf.gather(tensor, indices, axis=-1, batch_dims=1)


def _lengths_from_ids(ids, end_id):
    """Compute sequence lengths from word ids."""
    lengths = tf.not_equal(ids, end_id)
    lengths = tf.cast(lengths, tf.int32)
    lengths = tf.reduce_sum(lengths, axis=-1)
    return lengths
