# Parameters

Run parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference. It uses the following layout:

```yaml
data:
  # Data configuration (training and evaluation files, vocabularies, alignments, etc.)
params:
  # Training and inference hyperparameters (learning rate, optimizer, beam size, etc.)
train:
  # Training specific configuration (checkpoint frequency, number of training step, etc.)
eval:
  # Evaluation specific configuration (evaluation frequency, scorers, etc.)
infer:
  # Inference specific configuration (output scores, alignments, etc.)
score:
  # Scoring specific configuration
```

## Automatic configuration

Predefined models declare default parameters that should give solid performance out of the box. To enable automatic configuration, use the `--auto_config` flag:

```bash
onmt-main --model_type Transformer --config my_data.yml --auto_config train
```

The user provided `my_data.yml` file will minimally require the data configuration (see [Quickstart](quickstart.md) for example). You might want to also configure checkpoint related settings, the logging frequency, and the number of training steps.

At the start of the training, the configuration values actually used will be logged. If you want to change some of them, simply add the parameter in your configuration file to override the default value.

**Note:** default training values usually assume GPUs with at least 8GB of memory and a large system memory:

* If you encounter GPU out of memory issues, try overriding `batch_size` to a lower value.
* If you encounter CPU out of memory issues, try overriding `sample_buffer_size` to a fixed value.

## Multiple configuration files

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```bash
onmt-main --config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml [...]
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

If you are unsure about the configuration that is actually used or simply prefer working with a single file, consider using the `merge_config` script:

```bash
onmt-merge-config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml > config/my_config.yml
```

## Reference

Below is an exhaustive and documented configuration. **You should NOT copy and use this configuration, instead you should only define the parameters that you need.**

```yaml
# (optional) The directory where models and summaries will be saved.
# Can also be set with the command line option --model_dir.
# The directory is created if it does not exist.
model_dir: toy-ende

# (optional) Enable automatic parameters based on the selected model.
# Can also be set with the command line option --auto_config.
auto_config: true

data:
  # (required for train run type).
  train_features_file: data/toy-ende/src-train.txt
  train_labels_file: data/toy-ende/tgt-train.txt

  # (optional) A list with the weights of each training files, if multiple training
  # files were configured (default: null).
  train_files_weights: null

  # (optional) Pharaoh alignments of the training files.
  train_alignments: data/toy-ende/alignments-train.txt

  # (optional) File containing the weight of each example (one weight per line).
  # The loss value of each example is multiplied by its corresponding weight.
  example_weights: data/toy-ende/weights-train.txt

  # (required for train_end_eval and eval run types).
  eval_features_file: data/toy-ende/src-val.txt
  eval_labels_file: data/toy-ende/tgt-val.txt

  # (optional) Models may require additional resource files (e.g. vocabularies).
  source_vocabulary: data/toy-ende/src-vocab.txt
  target_vocabulary: data/toy-ende/tgt-vocab.txt

  # (optional) During export save the vocabularies as model assets, otherwise embed
  # them in the graph itself (default: true).
  export_vocabulary_assets: true

  # (optional) Tokenization configuration (or path to a configuration file).
  # See also: https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md
  source_tokenization:
    type: OpenNMTTokenizer
    params:
      mode: aggressive
      joiner_annotate: true
      segment_numbers: true
      segment_alphabet_change: true
  target_tokenization: config/tokenization/aggressive.yml

  # (optional) Pretrained embedding configuration.
  source_embedding:
    path: data/glove/glove-100000.txt
    with_header: true
    case_insensitive: true
    trainable: false

  # (optional) For language models, configure sequence control tokens (usually
  # represented as <s> and </s>). For example, enabling "start" and disabling "end"
  # allows nonconditional and unbounded generation (default: start=false, end=true).
  #
  # Advanced users could also configure this parameter for seq2seq models with e.g.
  # source_sequence_controls and target_sequence_controls.
  sequence_controls:
    start: false
    end: true

  # (optional) For sequence tagging tasks, the tagging scheme that is used (e.g. BIOES).
  # For supported schemes, additional evaluation metrics could be computed such as
  # precision, recall, etc. (accepted values: bioes; default: null).
  tagging_scheme: bioes

# Model and optimization parameters.
params:
  # The optimizer class name in tf.keras.optimizers or tfa.optimizers.
  optimizer: Adam
  # (optional) Additional optimizer parameters as defined in their documentation.
  # If weight_decay is set, the optimizer will be extended with decoupled weight decay.
  optimizer_params:
    beta_1: 0.8
    beta_2: 0.998
  learning_rate: 1.0

  # (optional) If set, overrides all dropout values configured in the model definition.
  dropout: 0.3

  # (optional) List of layer to not optimize.
  freeze_layers:
    - "encoder/layers/0"
    - "decoder/output_layer"

  # (optional) Weights regularization penalty (default: null).
  regularization:
    type: l2  # can be "l1", "l2", "l1_l2" (case-insensitive).
    scale: 1e-4  # if using "l1_l2" regularization, this should be a YAML list.

  # (optional) Average loss in the time dimension in addition to the batch dimension
  # (default: true when using "tokens" batch type, false otherwise).
  average_loss_in_time: false
  # (optional) High training loss values considered as outliers will be masked (default: false).
  mask_loss_outliers: false

  # (optional) The type of learning rate decay (default: null). See:
  #  * https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules
  #  * https://opennmt.net/OpenNMT-tf/package/opennmt.schedules.html
  # This value may change the semantics of other decay options. See the documentation
  # or the code.
  decay_type: NoamDecay
  # (optional unless decay_type is set) Decay parameters.
  decay_params:
    model_dim: 512
    warmup_steps: 4000
  # (optional) The number of training steps that make 1 decay step (default: 1).
  decay_step_duration: 1
  # (optional) After how many steps to start the decay (default: 0).
  start_decay_steps: 50000

  # (optional) The learning rate minimum value (default: 0).
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
  # (see https://opennmt.net/OpenNMT-tf/package/opennmt.data.noise.html)
  decoding_noise:
    - dropout: 0.1
    - replacement: [0.1, ｟unk｠]
    - permutation: 3
  # (optional) Define the subword marker. This is useful to apply noise at the
  # word level instead of the subword level (default: ￭).
  decoding_subword_token: ￭
  # (optional) Whether decoding_subword_token is used as a spacer (as in SentencePiece)
  # or a joiner (as in BPE).
  # If unspecified, will infer  directly from decoding_subword_token.
  decoding_subword_token_is_spacer: false
  # (optional) Minimum length of decoded sequences, end token excluded (default: 0).
  minimum_decoding_length: 0
  # (optional) Maximum length of decoded sequences, end token excluded (default: 250).
  maximum_decoding_length: 250

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
  # (optional) Size of output on an exported TensorFlow Lite model
  tflite_output_size: 250


# Training options.
train:
  # (optional) Training batch size. If set to 0, the training will search the largest
  # possible batch size.
  batch_size: 64
  # (optional) Batch size is the number of "examples" or "tokens" (default: "examples").
  batch_type: examples
  # (optional) Tune gradient accumulation to train with at least this effective batch size
  # (default: null).
  effective_batch_size: 25000

  # (optional) Save a checkpoint every this many steps (default: 5000).
  save_checkpoints_steps: null
  # (optional) How many checkpoints to keep on disk.
  keep_checkpoint_max: 3

  # (optional) Dump summaries and logs every this many steps (default: 100).
  save_summary_steps: 100

  # (optional) Maximum training step. If not set, train forever.
  max_step: 1000000
  # (optional) If true, makes a single pass over the training data (default: false).
  single_pass: false

  # (optional) The maximum length of feature sequences during training (default: null).
  maximum_features_length: 70
  # (optional) The maximum length of label sequences during training (default: null).
  maximum_labels_length: 70

  # (optional) The width of the length buckets to select batch candidates from.
  # A smaller value means less padding and increased efficiency. (default: 1).
  length_bucket_width: 1

  # (optional) The number of elements from which to sample during shuffling (default: 500000).
  # Set 0 or null to disable shuffling, -1 to match the number of training examples.
  sample_buffer_size: 500000

  # (optional) Moving average decay. Reasonable values are close to 1, e.g. 0.9999, see
  # https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
  # (default: null)
  moving_average_decay: 0.9999
  # (optional) Number of checkpoints to average at the end of the training to the directory
  # model_dir/avg (default: 0).
  average_last_checkpoints: 8


# (optional) Evaluation options.
eval:
  # (optional) The batch size to use (default: 32).
  batch_size: 30
  # (optional) Batch size is the number of "examples" or "tokens" (default: "examples").
  batch_type: examples

  # (optional) Evaluate every this many steps (default: 5000).
  steps: 5000

  # (optional) Save evaluation predictions in model_dir/eval/.
  save_eval_predictions: false
  # (optional) Scorer or list of scorers that are called on the saved evaluation
  # predictions.
  # Available scorers: bleu, rouge, wer, ter, prf
  scorers: bleu

  # (optional) The width of the length buckets to select batch candidates from.
  # If set, the eval data will be sorted by length to increase the translation
  # efficiency. The predictions will still be outputted in order as they are
  # available (default: 0).
  length_bucket_width: 5

  # (optional) Export a model when a metric has the best value so far (default: null).
  export_on_best: bleu
  # (optional) Format of the exported model (can be: "saved_model, "checkpoint",
  # "ctranslate2", "ctranslate2_int8", "ctranslate2_int16", "ctranslate2_float16",
  # default: "saved_model").
  export_format: saved_model
  # (optional) Maximum number of exports to keep on disk (default: 5).
  max_exports_to_keep: 5

  # (optional) Early stopping condition.
  # Should be read as: stop the training if "metric" did not improve more
  # than "min_improvement" in the last "steps" evaluations.
  early_stopping:
    # (optional) The target metric name (default: "loss").
    metric: bleu
    # (optional) The metric should improve at least by this much to be considered
    # as an improvement (default: 0)
    min_improvement: 0.01
    steps: 4

# (optional) Inference options.
infer:
  # (optional) The batch size to use (default: 16).
  batch_size: 10
  # (optional) Batch size is the number of "examples" or "tokens" (default: "examples").
  batch_type: examples

  # (optional) For compatible models, the number of hypotheses to output (default: 1).
  # This sets the parameter params/num_hypotheses.
  n_best: 1
  # (optional) For compatible models, also output the score (default: false).
  with_scores: false
  # (optional) For compatible models, also output the alignments
  # (can be: null, hard, soft, default: null).
  with_alignments: null

  # (optional) The width of the length buckets to select batch candidates from.
  # If set, the test data will be sorted by length to increase the translation
  # efficiency. The predictions will still be outputted in order as they are
  # available (default: 0).
  length_bucket_width: 5


# (optional) Scoring options.
score:
  # (optional) The batch size to use (default: 64).
  batch_size: 64
  # (optional) Batch size is the number of "examples" or "tokens" (default: "examples").
  batch_type: examples

  # (optional) The width of the length buckets to select batch candidates from.
  # If set, the input file will be sorted by length to increase efficiency.
  # The result will still be outputted in order as they are available (default: 0).
  length_bucket_width: 0

  # (optional) Also report token-level cross entropy.
  with_token_level: false
  # (optional) Also output the alignments (can be: null, hard, soft, default: null).
  with_alignments: null
```
