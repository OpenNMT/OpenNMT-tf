# Parameters

Run parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference. It uses the following layout:

```yaml
model_dir: path_to_the_model_directory

data:
  # Data configuration (training and evaluation files, vocabularies, alignments, etc.)
params:
  # Training and inference hyperparameters (learning rate, optimizer, beam size, etc.)
train:
  # Training specific configuration (checkpoint frequency, number of training step, etc.)
eval:
  # Evaluation specific configuration (evaluation frequency, external evaluators.)
infer:
  # Inference specific configuration (output scores, alignments, etc.)
score:
  # Scoring specific configuration
```

Here is an exhaustive and documented configuration:

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

  # (optional) For sequence tagging tasks, the tagging scheme that is used (e.g. BIOES).
  # For supported schemes, additional evaluation metrics could be computed such as
  # precision, recall, etc. (accepted values: bioes; default: null).
  tagging_scheme: bioes

# Model and optimization parameters.
params:
  # The optimizer class name in tf.train, tf.contrib.opt, or opennmt.optimizers.
  optimizer: AdamOptimizer
  # (optional) Additional optimizer parameters as defined in their documentation.
  optimizer_params:
    beta1: 0.8
    beta2: 0.998
  learning_rate: 1.0

  # (optional) Regexp or list of regexp matching variable names to not optimize.
  freeze_variables:
    - "transformer/encoder"

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

  # (optional) Horovod parameters.
  horovod:
    # (optional) Compression type for gradients (can be: "none", "fp16", default: "none").
    compression: none
    # (optional) Average the reduced gradients (default: false).
    average_gradients: false

  # (optional) Weights regularization penalty (default: null).
  regularization:
    type: l2  # can be "l1", "l2", "l1_l2" (case-insensitive).
    scale: 1e-4  # if using "l1_l2" regularization, this should be a YAML list.
  # (optional) Decoupled weight decay (default: null).
  weight_decay: 0.01

  # (optional) Average loss in the time dimension in addition to the batch dimension (default: False).
  average_loss_in_time: false

  # (optional) The type of learning rate decay (default: null). See:
  #  * https://www.tensorflow.org/versions/master/api_guides/python/train#Decaying_the_learning_rate
  #  * opennmt/utils/decay.py
  # This value may change the semantics of other decay options. See the documentation or the code.
  decay_type: exponential_decay
  # (optional unless decay_type is set) Decay parameters.
  decay_params:
    # The learning rate decay rate.
    decay_rate: 0.7
    # Decay every this many steps.
    decay_steps: 10000
    # (optional) If true, the learning rate is decayed in a staircase fashion (default: true).
    staircase: true
  # (optional) The number of training steps that make 1 decay step (default: 1).
  decay_step_duration: 1
  # (optional) After how many steps to start the decay (default: 0).
  start_decay_steps: 50000

  # (optional) The learning rate minimum value (default: 0).
  minimum_learning_rate: 0.0001
  # (optional) The learning rate maximum value (default: 1e6).
  maximum_learning_rate: 1e6

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
  # (optional) Number of hypotheses to return (default: 1). Set 0 to return all
  # available hypotheses. This value is also set by infer/n_best.
  num_hypotheses: 1
  # (optional) Length penaly weight to use during beam search (default: 0).
  length_penalty: 0.2
  # (optional) Coverage penaly weight to use during beam search (default: 0).
  coverage_penalty: 0.2
  # (optional) Sample predictions from the top K most likely tokens (requires
  # beam_width to 1). If 0, sample from the full output distribution (default: 1).
  sampling_topk: 1
  # (optional) High temperatures generate more random samples (default: 1).
  sampling_temperature: 1
  # (optional) Sequence of noise to apply to the decoding output. Each element
  # should be a noise type (can be: "dropout", "replacement", "permutation") and
  # the module arguments
  # (see http://opennmt.net/OpenNMT-tf/package/opennmt.layers.noise.html)
  decoding_noise:
    - dropout: 0.1
    - replacement: [0.1, ｟unk｠]
    - permutation: 3
  # (optional) Define the subword marker. This is useful to apply noise at the
  # word level instead of the subword level (default: ￭).
  decoding_subword_token: ￭
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

  # (optional) Enable contrastive learning mode, see
  # https://www.aclweb.org/anthology/P19-1623 (default: false).
  # See also "decoding_subword_token" that is used by this mode.
  contrastive_learning: false
  # (optional) The value of the parameter eta in the max-margin loss (default: 0.1).
  max_margin_eta: 0.1


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
  # This sets the parameter params/num_hypotheses.
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

## Automatic configuration

Predefined models declare default parameters that should give solid performance out of the box. To enable automatic configuration, use the `--auto_config` flag:

```bash
onmt-main train_and_eval --model_type Transformer --config my_data.yml --auto_config
```

The user provided `my_data.yml` file will minimaly require the data configuration. You might want to also configure checkpoint related settings, the logging frequency, and the number of training steps.

At the start of the training, the configuration values actually used will be logged. If you want to change some of them, simply add the parameter in your configuration file to override the default value.

**Note:** default training values usually assume GPUs with at least 8GB of memory and a large system memory:

* If you encounter GPU out of memory issues, try overriding `batch_size` to a lower value.
* If you encounter CPU out of memory issues, try overriding `sample_buffer_size` to a fixed value.

## Multiple configuration files

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```bash
onmt-main [...] --config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

If you are unsure about the configuration that is actually used or simply prefer working with a single file, consider using the `merge_config` script:

```bash
onmt-merge-config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml > config/my_config.yml
```

## TensorFlow session

The command line option `--session_config` can be used to configure the TensorFlow session that is created to execute TensorFlow graphs. The option takes a file containing a [`tf.ConfigProto`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto) message serialized in text format.

Here is an example to enable the `allow_growth` GPU option:

```text
$ cat config/session_config.txt
gpu_options {
  allow_growth: true
}
```

```bash
onmt-main [...] --session_config config/session_config.txt
```

For possible options and values, see the [`tf.ConfigProto`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto) file.
