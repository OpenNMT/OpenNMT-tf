# Reference: Configuration

```yaml
# The directory where models and summaries will be saved. It is created if it does not exist.
model_dir: toy-ende

data:
  # (required for train_and_eval and train run types).
  train_features_file: data/toy-ende/src-train.txt
  train_labels_file: data/toy-ende/tgt-train.txt

  # (optional) Pharaoh alignments of the training files.
  train_alignments: data/toy-ende/alignments-train.txt

  # (required for train_end_eval and eval run types).
  eval_features_file: data/toy-ende/src-val.txt
  eval_labels_file: data/toy-ende/tgt-val.txt

  # (optional) Models may require additional resource files (e.g. vocabularies).
  source_words_vocabulary: data/toy-ende/src-vocab.txt
  target_words_vocabulary: data/toy-ende/tgt-vocab.txt

  # (optional) OpenNMT tokenization configuration (or path to a configuration file).
  # See also: https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md
  source_tokenization:
    mode: aggressive
    joiner_annotate: true
    segment_numbers: true
    segment_alphabet_change: true
  target_tokenization: config/tokenization/aggressive.yml

  # (optional) Pretrained embedding configuration.
  source_embedding:
    path: data/glove/glove-100000.txt
    with_header: True
    case_insensitive: True
    trainable: False

# Model and optimization parameters.
params:
  # The optimizer class name in tf.train, tf.contrib.opt, or opennmt.optimizers.
  optimizer: AdamOptimizer
  # (optional) Additional optimizer parameters as defined in their documentation.
  optimizer_params:
    beta1: 0.8
    beta2: 0.998
  learning_rate: 1.0

  # (optional) Global parameter initialization [-param_init, param_init].
  param_init: 0.1

  # (optional) Maximum gradients norm (default: null).
  clip_gradients: 5.0
  # (optional) 1 training step will process this many batches and accumulates
  # their gradients (default: 1).
  gradients_accum: 1

  # (optional) For mixed precision training, the loss scaling to apply (a constant value or
  # an automatic scaling algorithm: "backoff", "logmax", default: "backoff")
  loss_scale: backoff
  # (optional) For mixed precision training, the additional parameters to pass the loss scale
  # (see the source file opennmt/optimizers/mixed_precision_wrapper.py).
  loss_scale_params:
    scale_min: 1.0
    step_factor: 2.0

  # (optional) Weights regularization penalty (default: null).
  regularization:
    type: l2  # can be "l1", "l2", "l1_l2" (case-insensitive).
    scale: 1e-4  # if using "l1_l2" regularization, this should be a YAML list.

  # (optional) Average loss in the time dimension in addition to the batch dimension (default: False).
  average_loss_in_time: false

  # (optional) The type of learning rate decay (default: null). See:
  #  * https://www.tensorflow.org/versions/master/api_guides/python/train#Decaying_the_learning_rate
  #  * opennmt/utils/decay.py
  # This value may change the semantics of other decay options. See the documentation or the code.
  decay_type: exponential_decay
  # (optional unless decay_type is set) The learning rate decay rate.
  decay_rate: 0.7
  # (optional unless decay_type is set) Decay every this many steps.
  decay_steps: 10000
  # (optional) The number of training steps that make 1 decay step (default: 1).
  decay_step_duration: 1
  # (optional) If true, the learning rate is decayed in a staircase fashion (default: True).
  staircase: true
  # (optional) After how many steps to start the decay (default: 0).
  start_decay_steps: 50000

  # (optional) Stop decay when this learning rate value is reached (default: 0).
  minimum_learning_rate: 0.0001

  # (optional) Type of scheduled sampling (can be "constant", "linear", "exponential",
  # or "inverse_sigmoid", default: "constant").
  scheduled_sampling_type: constant
  # (optional) Probability to read directly from the inputs instead of sampling categorically
  # from the output ids (default: 1).
  scheduled_sampling_read_probability: 1
  # (optional unless scheduled_sampling_type is set) The constant k of the schedule.
  scheduled_sampling_k: 0

  # (optional) The label smoothing value.
  label_smoothing: 0.1

  # (optional) Width of the beam search (default: 1).
  beam_width: 5
  # (optional) Length penaly weight to apply on hypotheses (default: 0).
  length_penalty: 0.2
  # (optional) Sample predictions from the top K most likely tokens (requires
  # beam_width to 1). If 0, sample from the full output distribution (default: 1).
  sampling_topk: 1
  # (optional) Minimum length of decoded sequences, end token excluded (default: 0).
  minimum_decoding_length: 0
  # (optional) Maximum decoding iterations before stopping (default: 250).
  maximum_iterations: 200

  # (optional) Replace unknown target tokens by the original source token with the
  # highest attention (default: false).
  replace_unknown_target: false

  # (optional) The type of guided alignment cost to compute (can be: "null", "ce", "mse",
  # default: "null").
  guided_alignment_type: null
  # (optional) The weight of the guided alignment cost (default: 1).
  guided_alignment_weight: 1


# Training options.
train:
  # (optional when batch_type=tokens) If not set, the training will search the largest
  # possible batch size.
  batch_size: 64
  # (optional) Batch size is the number of "examples" or "tokens" (default: "examples").
  batch_type: examples
  # (optional) Tune gradient accumulation to train with at least this effective batch size
  # (default: null).
  effective_batch_size: 25000

  # (optional) Save a checkpoint every this many steps (default: null). Can not be
  # specified with save_checkpoints_secs.
  save_checkpoints_steps: null
  # (optional) Save a checkpoint every this many seconds (default: 600). Can not be
  # specified with save_checkpoints_steps.
  save_checkpoints_secs: 600
  # (optional) How many checkpoints to keep on disk.
  keep_checkpoint_max: 3

  # (optional) Dump summaries and logs every this many steps (default: 100).
  save_summary_steps: 100

  # (optional) Train for this many steps. If not set, train forever.
  train_steps: 1000000
  # (optional) If true, makes a single pass over the training data (default: false).
  single_pass: false

  # (optional) The maximum length of feature sequences during training (default: null).
  maximum_features_length: 70
  # (optional) The maximum length of label sequences during training (default: null).
  maximum_labels_length: 70

  # (optional) The width of the length buckets to select batch candidates from (default: 5).
  bucket_width: 5

  # (optional) The number of threads to use for processing data in parallel (default: 4).
  num_threads: 4
  # (optional) The number of batches to prefetch asynchronously. If not set, use an
  # automatically tuned value on TensorFlow 1.8+ and 1 on older versions. (default: null).
  prefetch_buffer_size: null

  # (optional) The number of elements from which to sample during shuffling (default: 500000).
  # Set 0 or null to disable shuffling, -1 to match the number of training examples.
  sample_buffer_size: 500000

  # (optional) Number of checkpoints to average at the end of the training to the directory
  # model_dir/avg (default: 0).
  average_last_checkpoints: 8


# (optional) Evaluation options.
eval:
  # (optional) The batch size to use (default: 32).
  batch_size: 30

  # (optional) The number of threads to use for processing data in parallel (default: 1).
  num_threads: 1
  # (optional) The number of batches to prefetch asynchronously (default: 1).
  prefetch_buffer_size: 1

  # (optional) Evaluate every this many seconds (default: 18000).
  eval_delay: 7200

  # (optional) Save evaluation predictions in model_dir/eval/.
  save_eval_predictions: false
  # (optional) Evalutator or list of evaluators that are called on the saved evaluation predictions.
  # Available evaluators: sacreBLEU, BLEU, BLEU-detok, ROUGE
  external_evaluators: sacreBLEU

  # (optional) Model exporter(s) to use during the training and evaluation loop:
  # last, final, best, or null (default: last).
  exporters: last


# (optional) Inference options.
infer:
  # (optional) The batch size to use (default: 1).
  batch_size: 10

  # (optional) The width of the length buckets to select batch candidates from.
  # If set, the test data will be sorted by length to increase the translation
  # efficiency. The predictions will still be outputted in order as they are
  # available (default: 0).
  bucket_width: 5

  # (optional) The number of threads to use for processing data in parallel (default: 1).
  num_threads: 1
  # (optional) The number of batches to prefetch asynchronously (default: 1).
  prefetch_buffer_size: 1

  # (optional) For compatible models, the number of hypotheses to output (default: 1).
  n_best: 1
  # (optional) For compatible models, also output the score (default: false).
  with_scores: false
  # (optional) For compatible models, also output the alignments (can be: "null", "hard",
  # default: "null").
  with_alignments: null


# (optional) Scoring options.
score:
  # (optional) The batch size to use (default: 64).
  batch_size: 64
  # (optional) The number of threads to use for processing data in parallel (default: 1).
  num_threads: 1
  # (optional) The number of batches to prefetch asynchronously (default: 1).
  prefetch_buffer_size: 1

  # (optional) Also report token-level cross entropy.
  with_token_level: false
  # (optional) Also output the alignments (can be: "null", "hard", default: "null").
  with_alignments: null
```
