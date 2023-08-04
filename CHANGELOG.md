**Notes on versioning**

OpenNMT-tf follows [semantic versioning 2.0.0](https://semver.org/). The API covers:

* command line options
* configuration files
* checkpoints of non experimental models
* classes and functions documented on the [online documentation](https://opennmt.net/OpenNMT-tf/package/overview.html) and directly accessible from the top-level `opennmt` package

---

## [Unreleased]

### New features

### Fixes and improvements

## [2.32.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.32.0) (2023-08-04)

### New features

* Support TensorFlow 2.12 and 2.13
* Make timeout value configurable while searching for an optimal batch size

## [2.31.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.31.0) (2023-01-13)

### New features

* Add option `--jit_compile` to compile the model with XLA (only applied in training at the moment)

### Fixes and improvements

* Improve correctness of gradient accumulation and multi-GPU training by normalizing the gradients with the true global batch size instead of using an approximation
* Report the total number of tokens per second in the training logs, in addition to the source and target numbers
* Relax the sacreBLEU version requirement to include any 2.x versions

## [2.30.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.30.0) (2022-12-12)

### Changes

* The model attribute `ctranslate2_spec` has been removed as it is no longer relevant with the new CTranslate2 converter
* The global gradient norm is no longer reported in TensorBoard because it was misleading: it did not take into account gradient accumulation and multi-GPU

### New features

* Support TensorFlow 2.11 (note that the new Keras optimizers are not yet supported, if you are creating optimizers manually please use an optimizer in `tf.keras.optimizers.legacy` for now)
* Support CTranslate2 3.0
* Add training parameter `pad_to_bucket_boundary` to pad the batch length to a multiple of `length_bucket_width` (this is useful to reduce the number of recompilation with XLA)
* Integrate the scorers `chrf` and `chrf++` from SacreBLEU

### Fixes and improvements

* Fix error when training with Horovod and using an early stopping condition
* Fix error when using guided alignment with mixed precision

## [2.29.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.29.1) (2022-10-03)

### Fixes and improvements

* Fix error when using gzipped training data files
* Remove unnecessary casting in `MultiHeadAttention` for a small performance improvement

## [2.29.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.29.0) (2022-09-26)

### New features

* Support TensorFlow 2.10
* Add model configurations `ScalingNmtEnDe` and `ScalingNmtEnFr` from [Ott et al. 2018](https://aclanthology.org/W18-6301/)
* Add embedding parameter `EmbeddingsSharingLevel.AUTO` to automatically share embeddings when the vocabulary is shared
* Extend method `Runner.average_checkpoints` to accept a list of checkpoints to average

### Fixes and improvements

* Make batch size autotuning faster when using gradient accumulation

## [2.28.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.28.0) (2022-07-29)

### New features

* Add `initial_learning_rate` parameter to the `InvSqrtDecay` schedule
* Add new arguments to the `Transformer` constructor:
  * `mha_bias`: to disable bias terms in the multi-head attention (as presented in the original paper)
  * `output_layer_bias`: to disable bias in the output linear layer

### Fixes and improvements

* Fix incorrect dtype for `SequenceRecordInputter` length vector
* Fix rounding error when batching datasets which could make the number of tokens in a batch greater than the configured batch size
* Fix deprecation warning when using `distutils.version.LooseVersion`, use `packaging.version.Version` instead
* Make the length dimension unknown in the dataset used for batch size autotuning so that it matches the behavior in training
* Update SacreBLEU requirement to include new version 2.2

## [2.27.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.27.1) (2022-06-02)

### Fixes and improvements

* Fix evaluation and scoring with language models

## [2.27.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.27.0) (2022-05-30)

### Changes

* Remove support for older TensorFlow versions 2.4 and 2.5
* Remove support for deprecated Python version 3.6

### New features

* Support TensorFlow 2.9
* Integrate the new CTranslate2 converter to export more Transformer variants, including multi-features models

### Fixes and improvements

* Fix error when loading the SavedModel of Transformer models with relative position representations
* Fix dataset error in inference with language models
* Fix batch size autotuning error with language models
* Fix division by zero error on some systems when the time to the last training log is too small

## [2.26.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.26.1) (2022-03-31)

### Fixes and improvements

* Fix documentation build error

## [2.26.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.26.0) (2022-03-31)

### New features

* Add learning rate schedule `InvSqrtDecay`
* Enable CTranslate2 conversion for models using GELU or Swish activations

### Fixes and improvements

* Fix inference error when using the `decoding_noise` parameter
* Clarify the inference log about buffered predictions

## [2.25.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.25.0) (2022-02-21)

### New features

* Support TensorFlow 2.8
* Add training flag `--continue_from_checkpoint` to simplify continuing the training in another model directory (to be used in combination with `--checkpoint_path`)

### Fixes and improvements

* Fix target unknowns replacement when the source has BOS or EOS tokens
* Update length constraints in Transformer automatic configuration to work with multiple sources
* Allow explicit configuration of the first argument of learning rate schedules (if not set, `learning_rate` is passed as the first argument)

## [2.24.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.24.0) (2021-12-17)

### New features

* Add experimental parameter `mask_loss_outliers` to mask high loss values considered as outliers (requires the `tensorflow-probability` module)

### Fixes and improvements

* Fix TensorFlow Lite conversion for models using a `PositionEmbedder` layer
* Automatically pad the weights of linear layers to enable Tensor Cores in mixed precision training
* Correctly set the CTranslate2 options `alignment_layer` and `alignment_heads` when converting models using the attention reduction `AVERAGE_LAST_LAYER`
* Raise an error if a training dataset or annotation file has an unexpected size
* Warn about duplicated tokens when loading vocabularies

## [2.23.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.23.0) (2021-11-15)

### Changes

* Remove support for TensorFlow 2.3

### New features

* Support TensorFlow 2.7
* Add CTranslate2 exporter with "int8_float16" quantization

### Fixes and improvements

* Improve performance when applying the OpenNMT tokenization during training by vectorizing the dataset transformation
* Disable configuration merge for fields `optimizer_params` and `decay_params`
* Enable the CTranslate2 integration when installing OpenNMT-tf on Windows
* Include PyYAML 6 in supported versions

## [2.22.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.22.0) (2021-09-30)

### New features

* Support TensorFlow Lite conversion for Transformer models
* Make the options `model_dir` and `auto_config` available in both the command line and the configuration file
* Paths in the `data` configuration can now be relative to the model directory

### Fixes and improvements

* Fix encoding when writing sentences detokenized by an in-graph tokenizer
* Always output the tokenized target in scoring even when a target tokenization is configured
* Enable the OpenNMT Tokenizer when installing OpenNMT-tf on Windows

## [2.21.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.21.0) (2021-08-30)

### New features

* Support TensorFlow 2.6
* Add tokenizer `SentencePieceTokenizer`, an in-graph SentencePiece tokenizer provided by [tensorflow-text](https://www.tensorflow.org/text)
* Add methods to facilitate training a `Model` instance:
  * `model.compute_training_loss`
  * `model.compute_gradients`
  * `model.train`
* Add `--output_file` argument to `score` command

### Fixes and improvements

* Fix `make_features` method of inputters `WordEmbedder` and `SequenceRecordInputter` to work on a batch of elements
* Fix error when `SelfAttentionDecoder` is called without `memory_sequence_length`
* Fix `ConvEncoder` on variable-length inputs
* Support SacreBLEU 2.0

## [2.20.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.20.1) (2021-07-01)

### Fixes and improvements

* Fix missing environment variables in the child process when autotuning the batch size
* Fix error during evaluation when setting the inference parameter `n_best` > 1
* Fix error in Python serving example when using TensorFlow 2.5
* Log some information about the input layer after initialization (vocabulary size, special tokens, etc.)
* Update the minimum required pyonmttok version to 1.26.4 to include the latest fixes

## [2.20.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.20.0) (2021-06-17)

### New features

* Update the minimum required CTranslate2 version to 2.0

### Fixes and improvements

* Set a timeout for each training attempt when autotuning the batch size
* Set `keep_checkpoint_max` to `average_last_checkpoints` if the later value is larger
* Update the minimum required pyonmttok version to 1.26.2 to include the latest fixes

## [2.19.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.19.0) (2021-05-31)

### New features

* Support TensorFlow 2.5

### Fixes and improvements

* Fix dtype error in RNN decoder when enabling mixed precision
* Pass training flag to tokenizers to disable subword regularization in inference
* Update Sphinx from 2.3 to 3.5 to generate the documentation

## [2.18.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.18.1) (2021-04-27)

### Fixes and improvements

* Fix vocabulary update for models with shared embeddings
* Fix a compatibility issue with TensorFlow 2.5 for early users
* When all training attempts fail in batch size autotuning, log the error message of the last attempt

## [2.18.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.18.0) (2021-04-19)

### New features

* Add `TransformerBaseSharedEmbeddings` and `TransformerBigSharedEmbeddings` in the model catalog

### Fixes and improvements

* Fix loss normalization when using sentence weighting
* Tune the automatic batch size selection to avoid some out of memory errors
* Harmonize training logs format when using `onmt-main`

## [2.17.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.17.1) (2021-03-23)

### Fixes and improvements

* Fix some cases where batch size autotuning failed
* Raise an explicit error when batch size autotuning failed

## [2.17.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.17.0) (2021-03-15)

### Changes

* The `tensorflow` package is no longer a direct dependency of OpenNMT-tf and should be installed separately. This is to facilitate support of other TensorFlow packages such as `intel-tensorflow`, `tensorflow-rocm`, etc. Use `pip install OpenNMT-tf[tensorflow]` to keep the previous behavior.

### New features

* Use the [Keras mixed precision API](https://www.tensorflow.org/guide/mixed_precision) instead of the `auto_mixed_precision` graph optimization pass
* Support mixed precision training in eager execution (for debugging)
* Support batch size autotuning with the "examples" batch type

### Fixes and improvements

* Fix some TensorFlow Lite conversion errors
* Improve batch size autotuning accuracy by building batches with the maximum sequence lengths
* Reduce batch size autotuning time by running less iterations and disabling dataset buffering
* Scale the loss instead of the gradients when aggregating multiple batches
* Address some deprecation warnings in TensorFlow 2.4

## [2.16.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.16.0) (2021-02-25)

### Changes (non breaking)

* Rename evaluation parameter `external_evaluators` to `scorers`
* Rename export command line option `--export_dir` to `--output_dir`
* Rename export command line option `--export_format` to `--format`

### New features

* Add TensorFlow Lite exporters `tflite` and `tflite_float16` (only RNN-based sequence to sequence models are currently supported)
* Add argument `pre_norm` in the Transformer constructor to disable the pre-norm architecture and use the post-norm architecture as described in the [original paper](https://arxiv.org/abs/1706.03762)
* Add argument `attention_reduction` in the Transformer constructor to define how multi-head attention matrices are reduced into a single attention matrix
* Support batch type "tokens" and length bucketing in score task
* Support initializing the decoder output layer from its constructor (until now a separate call to `decoder.initialize(...)` was required)

### Fixes and improvements

* Fix error on 0-length inputs in `MultiHeadAttention` with relative position representations
* Fix CTranslate2 export for models that enable BOS or EOS tokens on the source side (see `source_sequence_controls` data parameter)
* Load YAML configuration with the `safe_load` function to prevent arbitrary code execution

## [2.15.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.15.0) (2021-01-28)

### New features

* Add sentence weighting: the data parameter `example_weights` accepts a file with one weight per line that are used to scale the loss of the corresponding training example
* Summarize out of vocabulary tokens at the end of the training logs:
  * ratio of OOV tokens in the training data
  * 10 most frequent OOV tokens
* [API] Add argument `fallback_to_cpu` to `Runner.train` to declare whether CPU training is allowed or not (defaults to `True` for backward compatibility)

### Fixes and improvements

* Fix error when computing BLEU score with SacreBLEU
* Fix vocabulary generation when using SentencePiece with a pre-tokenization
* Remove verbose checkpoint warnings when an exception occurs before checkpoint weights are actually loaded
* Enable the `pyonmttok` and `ctranslate2` dependencies on macOS
* Reformat the entire codebase with [Black](https://github.com/psf/black)

## [2.14.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.14.0) (2020-12-28)

### New features

* Support pre-tokenization when training a SentencePiece model and vocabulary with `onmt-build-vocab`
* Add quantized CTranslate2 exporters: `ctranslate2_int8`, `ctranslate2_int16`, `ctranslate2_float16`
* Re-add `decay_step_duration` training parameter from OpenNMT-tf V1 to delay the updates of learning rate schedules

### Fixes and improvements

* Fix error when training with multiple GPUs and TensorFlow 2.4
* Fix error when setting a Keras mixed precision policy and using high-level APIs such as `Runner` or `Trainer`
* Fix error when computing the dataset size and using a `MixedInputter`
* Fix the Python wheel pushed to PyPI that declared Python 2 compatibility
* Return the learning rate and the losses as a float values and not tensors in the training summary
* Remove `pyter3` dependency and compute TER with SacreBleu
* Remove unclear warning when setting `--model` or `--model_type` and a checkpoint already exists
* Raise error when training a SentencePiece model and vocabulary with `onmt-build-vocab` and using `--min_frequency` which is incompatible

## [2.13.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.13.0) (2020-10-20)

### New features

* [API] `opennmt.Encoder` now accepts `tf.RaggedTensor` as input
* [API] `opennmt.Runner.train` can return a summary of the training (e.g. average loss, last step, etc.)
* [API] Add function `opennmt.data.create_lookup_tables` to create `tf.lookup` tables from a vocabulary file

### Fixes and improvements

* Raise errors (not warnings) when starting the training but end conditions are met
* Ignore non training time when computing training throughput: ignore initialization, evaluation, and checkpoint saving
* Print OpenNMT-tf version in the training logs
* Improve error message when `model.initialize()` is not called

## [2.12.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.12.1) (2020-09-16)

### Fixes and improvements

* Update sacrebleu to 1.4.14 to fix Python 3.5 compatibility

## [2.12.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.12.0) (2020-08-31)

### New features

* Update to TensorFlow 2.3 and TensorFlow Addons 0.11
* Support running `onmt-main` with eager execution for development or debugging (see `--eager_execution` command line argument)

### Fixes and improvements

* Fix `--data_dir` prefix when the `data` block contains non string values
* Replace usage of the custom op `Addons>GatherTree` in beam search with a pure TensorFlow implementation to make model export and serving easier
* Raise an error when enabling `case_feature` for tokenization (unsupported in OpenNMT-tf)

## [2.11.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.11.1) (2020-06-25)

### Fixes and improvements

* Fix undefined gradient shape error in relative MultiHeadAttention
* Do not require `--labels_file` argument when evaluating language models

## [2.11.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.11.0) (2020-06-17)

### New features

* Dump a PyTorch-style description of the model in the logs
* Support bucketing the evaluation dataset by length for improved efficiency (parameter `length_bucket_width`, usually enabled by default with `--auto_config`)
* Accept "tokens" batch type for evaluation and inference (parameter `batch_type`)
* Accept passing a flat list of files on the command line for nested inputters
* Add "checkpoint" export format to export models as TensorFlow checkpoints
* Add `opennmt.layers.DenseReducer` to reduce inputs with a linear transformation
* Add function `opennmt.utils.average_checkpoints_into_layer` to average a list of checkpoints into an existing layer instance

### Fixes and improvements

* Improve training performance when not using gradient accumulation
* Simplify graph generated by `opennmt.layers.MultiplyReducer`
* Update the online documentation and add a list of direct children in each class reference, allowing to navigate up and down in the class inheritance

## [2.10.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.10.1) (2020-06-04)

### Fixes and improvements

* Fix error when running RNN models with `onmt-main`
* Add some missing functions in the online API documentation

## [2.10.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.10.0) (2020-05-28)

### New features

* Update to TensorFlow 2.2 and TensorFlow Addons 0.10
* Inputters can override the method `keep_for_training` to customize data filtering during the training

### Fixes and improvements

* Fix serving function for language models
* Fix possible crash when using the `prf` evaluator
* Reduce time to start the training by optimizing the computation of the training data size
* Add more logs during inference to indicate that progress is being made and the process is not "stuck"
* Update SacreBLEU to 1.4.9

## [2.9.3](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.9.3) (2020-05-06)

### Fixes and improvements

* Fix type error when online tokenization is called on an empty line

## [2.9.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.9.2) (2020-04-22)

### Fixes and improvements

* Pin `sacrebleu` package to version 1.4.4 to fix installation issue on Windows
* Clarify error when the training dataset is empty

## [2.9.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.9.1) (2020-04-14)

### Fixes and improvements

* Fix error in onmt-main when using run types other than train

## [2.9.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.9.0) (2020-04-07)

### New features

* Horovod support with the training flag `--horovod`
* New external scorers: Word Error Rate (`wer`), Translation Error Rate (`ter`), and Precision-Recall-FMeasure (`prf`)
* Evaluation parameter `max_exports_to_keep` to limit the number of exported models

### Fixes and improvements

* Do not report "target words/s" when training language models

## [2.8.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.8.1) (2020-03-24)

### Fixes and improvements

* Disable dropout in layers that are frozen by the `freeze_layers` parameter
* Fix batch size autotuning that ignored the mixed precision flag
* Fix sparse gradients that were unnecessarily converted to dense gradients in mixed precision training
* Only compute the gradients global norm every `save_summary_steps` steps to save a few computation during training
* Simplify some ops in the inference graph

## [2.8.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.8.0) (2020-03-02)

### New features

* Allow setting different number of encoder/decoder layers in `Transformer` constructor

### Fixes and improvements

* Fix decoder initialization with multi source encoder
* Fix command line parsing when `--config` is used before the run type
* Log more information on length mismatch when parsing alignments
* Log number of model parameters when starting the training

## [2.7.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.7.0) (2020-02-14)

### New features

* Enable CTranslate2 export for `TransformerBaseRelative` and `TransformerBigRelative` models
* Update TensorFlow Addons to 0.8

### Fixes and improvements

* Log the number of frozen weights when using the parameter `freeze_layers`
* More helpful error messages when layer names configured in `freeze_layers` are incorrect
* Improve beam search efficiency by avoiding some unecessary state reordering
* Add usage examples for symbols in `opennmt.data`

## [2.6.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.6.0) (2020-01-28)

### New features

* Multiple training files can be configured in `train_features_file`/`train_labels_file`, and `train_files_weights` optionally assign a weight to each file (see *Data* section in the documentation)
* Support exporting compatible models to CTranslate2 format (see `export_format` option)
* `moving_average_decay` training parameter to enable exponential moving average of the model variables

### Fixes and improvements

* Fix error when starting a language model training
* Use `tf.keras.layers.LayerNormalization` instead of custom implementation for improved efficiency
* Fix possible duplicate call to checkpoint saving at the end of the training
* Ignore BLEU evaluation warning when run on tokenized data (which also caused duplicated logs for the rest of the training)
* Improve accuracy of reported prediction time by ignoring the initial graph construction

## [2.5.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.5.1) (2020-01-20)

### Fixes and improvements

* Fix first value of steps per second metric when continuing a training
* Fix reporting of learning rate values with some optimizers

## [2.5.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.5.0) (2020-01-16)

### New features

* Update to TensorFlow 2.1<br/>OpenNMT-tf now depends on the `tensorflow` pip package instead of `tensorflow-gpu`. The `tensorflow` package now includes GPU support by default. If you are upgrading an existing environment, we recommend uninstalling the `tensorflow-gpu` package before doing so.
* Update to TensorFlow Addons 0.7 with Windows support
* Data parameter `export_vocabulary_assets` to control whether vocabularies are exported as file assets or embedded in the graph itself

### Fixes and improvements

* Fix error when reading loss returned by a sequence classifier model
* Fix error on sequences of length 1 in the sequence tagger with CRF
* Export vocabularies as file assets by default
* Remove an unnecessary synchronization when training multiple replicas
* Internal cleanup to fully remove Python 2 support

## [2.4.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.4.0) (2019-12-10)

### New features

* Transformer models with relative position representation: `TransformerRelative` and `TransformerBigRelative`

### Fixes and improvements

* Fix invalid iteration count in checkpoint after a vocabulary update
* Fix possible NaN loss when retraining after a vocabulary update
* Fix checkpoint averaging for models with custom variable names
* Update `opennmt.convert_to_v2_config` to not fail on a V2 configuration
* Change default value of `average_loss_in_time` based on `batch_type`
* Reuse the same Python interpreter when running batch size auto-tuning

## [2.3.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.3.0) (2019-11-25)

### New features

* Predefined models `NMTSmallV1`, `NMTMediumV1`, and `NMTBigV1` for compatibility with OpenNMT-tf v1
* Function `opennmt.convert_to_v2_config` to automatically upgrade a V1 configuration
* Function `opennmt.utils.is_v1_checkpoint` to detect OpenNMT-tf v1 checkpoints

### Fixes and improvements

* Fix error when using `auto_config` with model `LstmCnnCrfTagger`
* Fix incomplete `Model.create_variables` after manually calling `Model.build`
* Increase `LayerNorm` default epsilon value to be closer to TensorFlow and PyTorch defaults

## [2.2.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.2.1) (2019-11-07)

### Fixes and improvements

* Ensure that each training GPU receives a batch with the size configured by the user
* Fix error on the last partial batch when using multiple GPUs with `single_pass` enabled

## [2.2.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.2.0) (2019-11-06)

### New features

* Return detokenized predictions when using a in-graph tokenizer
* Injection of the special tokens `<s>` and `</s>` for language models can be configured with the data parameter `sequence_controls`

### Fixes and improvements

* Fix the batch size in multi GPU training that was not scaled by the number of devices
* When updating vocabularies, mirror the existing embeddings distribution for newly created embeddings
* Fix error when running `onmt-tokenize-text` and `onmt-detokenize-text` scripts
* Transformer decoder now always returns the attention on the first source
* Calling `model.initialize()` also initialize the decoder (if any)

## [2.1.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.1.1) (2019-10-18)

### Fixes and improvements

* Force tokenizers and noisers to run on CPU to avoid errors when placing strings on GPU
* Do not apply noise modules on empty inputs
* Fix training of `SequenceToSequence` models with guided alignment
* Fix training of `SequenceClassifier` models

## [2.1.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.1.0) (2019-10-10)

### New features

* `onmt-build-vocab` script can now train a SentencePiece model and vocabulary from raw data
* Enable automatic model export during evaluation with `export_on_best` parameter
* Add perplexity in evaluation metrics
* Extend tokenization configuration to support in-graph tokenizers (currently `SpaceTokenizer` and `CharacterTokenizer`)
* Parameter `decoder_subword_token_is_spacer` to configure the type of `decoder_subword_token`
* [API] Support `tf.RaggedTensor` in tokenizer API

### Fixes and improvements

* Fix early stopping logic
* Support spacer `decoder_subword_token` that is used as a suffix
* Improve errors when `--model` or `--model_type` options are invalid

## [2.0.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.0.1) (2019-10-04)

### Fixes and improvements

* Fix error when initializing language models
* Fix `eval` run when `--features_file` is not passed

## [2.0.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v2.0.0) (2019-10-01)

OpenNMT-tf 2.0 is the first major update of the project. The goal of this release is to use the new features and practices introduced by TensorFlow 2.0.

### Breaking changes

See the [2.0 Transition Guide](docs/v2_transition.md) for details about the following changes.

* TensorFlow 2.0 is now required
* Python 3.5 or greater is now required
* Checkpoints are no longer compatible as the code now uses object-based instead of name-based checkpointing (except Transformer checkpoints which are automatically upgraded when loaded)
* The `onmt-main` script now makes use of subparsers which require to move the run type and it specific options to the end of the command
* Some predefined models have been renamed or changed, see the [transition guide](docs/v2_transition.md#changed-predefined-models)
* Some parameters in the YAML configuration have been renamed or changed, see the [transition guide](docs/v2_transition.md#changed-parameters)
* A lot of public classes and functions have changed, see the [API documentation](http://opennmt.net/OpenNMT-tf/package/overview.html) for details
* TFRecord files generated with the `opennmt.inputters.write_sequence_record` function or the `onmt-ark-to-records` script are no longer compatible and should be re-generated

This version also changes the public API scope of the project:

* Only public symbols accessible from the top-level `opennmt` package and visible on the online documentation are now part of the public API and covered by backward compatibility guarantees
* The minimum required TensorFlow version is no longer part of the public API and can change in future minor versions

### New features

* Object-based layers extending `tf.keras.layers.Layer`
* Many new reusable modules and layers, see the [API documentation](http://opennmt.net/OpenNMT-tf/package/overview.html)
* Replace `tf.estimator` by custom loops for more control and clearer execution path
* Multi-GPU training with `tf.distribute`
* Support early stopping based on any evaluation metrics
* Support GZIP compressed datasets
* `eval` run type accepts `--features_file` and `--labels_file` to evaluate files other than the ones defined in the YAML configuration
* Accept `with_alignments: soft` to output soft alignments during inference or scoring
* `dropout` can be configured in the YAML configuration to override the model values

### Fixes and improvements

* Code and design simplification following TensorFlow 2.0 changes
* Improve logging during training
* Log level configuration also controls TensorFlow C++ logs
* All public classes and functions are now properly accessible from the root package `opennmt`
* Fix dtype error after updating the vocabulary of an averaged checkpoint
* When updating vocabularies, weights of new words are randomly initialized instead of zero initialized

### Missing features

Some features available in OpenNMT-tf v1 were removed or are temporarily missing in this v2 release. If you relied on some of them, please open an issue to track future support or find workarounds.

* Asynchronous distributed training
* Horovod integration
* Adafactor optimizer
* Global parameter initialization strategy (a Glorot/Xavier uniform initialization is used by default)
* Automatic SavedModel export on evaluation
* Average attention network

## [1.25.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.25.1) (2019-09-25)

### Fixes and improvements

* Fix language model decoding from context to pass the correct decoding step to the model
* Fix WordDropout module when the input contains a single word
* When updating vocabularies, weights of new words are randomly initialized instead of zero initialized

## [1.25.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.25.0) (2019-09-13)

### New features

* [Contrastive learning](https://www.aclweb.org/anthology/P19-1623) to reduce word omission errors

## [1.24.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.24.1) (2019-08-29)

### Fixes and improvements

* Fix NaN and inf values when using mixed precision and gradient clipping

## [1.24.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.24.0) (2019-06-26)

### New features

* Coverage penalty during beam search
* `freeze_variables` parameters to not update selected weights

### Fixes and improvements

* Improve length penalty correctness

## [1.23.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.23.1) (2019-06-07)

### Fixes and improvements

* Fix invalid offset in alignment vectors
* Remove debug print in noise module

## [1.23.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.23.0) (2019-05-30)

### New features

* Support applying noise on the decoding output ("beam+noise" strategy from [Edunov et al. 2018](https://arxiv.org/abs/1808.09381))

### Fixes and improvements

* Fix crash on TensorFlow 1.14 due to an [API change](https://github.com/tensorflow/tensorflow/issues/29024)
* Fix unexpected number of hypotheses returned by exported models: by default only the best is returned and the number should be configured by either `params/num_hypotheses` or `infer/n_best`
* Do not require setting an external evaluator when saving evaluation predictions
* Write external evaluators summaries in a separate directory to avoid interfering with BestExporter

## [1.22.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.22.2) (2019-05-17)

### Fixes and improvements

* Update PyYAML to 5.1 and silence warnings
* Reduce default visible memory for batch size auto-tuning to handle larger memory usage variations during training

## [1.22.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.22.1) (2019-04-29)

### Fixes and improvements

* Fix usage of `--checkpoint_path` during training
* Fix embedding sharing with a `ParallelInputter` as input

## [1.22.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.22.0) (2019-04-06)

### New features

* Unified dynamic decoding implementation
* Support random sampling during beam search

### Fixes and improvements

* More than 3x faster beam search decoding
* Improve correctness of alignment vectors for the other hypotheses

## [1.21.7](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.7) (2019-03-20)

### Fixes and improvements

* Fix error when sharing target embedding and softmax weights
* Run checkpoint utilities in a separate graph to avoid possible variables collision

## [1.21.6](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.6) (2019-03-12)

### Fixes and improvements

* Fix inputter initialization from the configuration file: fields such as `source_tokenization`, `target_embedding`, etc. were ignored in 1.21.0.

## [1.21.5](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.5) (2019-03-11)

### Fixes and improvements

* Fix compatibility issue with legacy TensorFlow 1.4
* Fix inference of language models
* Fix inference error when using `replace_unknown_target` and the alignment vector was empty

## [1.21.4](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.4) (2019-03-07)

### Fixes and improvements

* Fix error during manual model export

## [1.21.3](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.3) (2019-03-06)

### Fixes and improvements

* Fix multi GPU training: some variables were not correctly reused when building the graph for other devices

## [1.21.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.2) (2019-03-05)

### Fixes and improvements

* Fix checkpoint restore during evaluation when using a `PositionEmbedder` or a `DenseBridge` in the decoder
* Fix inputters parameter sharing: shared variables were overriden when the layer was invoked (requires retraining)

## [1.21.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.1) (2019-03-04)

### Fixes and improvements

* Allow configuring `tagging_scheme` in the data configuration
* Fix dimension mismatch when using `replace_unknown_target`

## [1.21.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.21.0) (2019-03-01)

### New features

* New experimental model type `LanguageModel` to train generative language models (see the example [GPT-2 configuration](https://github.com/OpenNMT/OpenNMT-tf/blob/master/config/models/gpt_2.py)). The usage is the same as a sequence to sequence model except that "labels" data should not be set.
* `cosine_annealing` learning rate decay
* `weight_decay` parameter to apply decoupled weight decay regularization (as described in [Loshchilov et al. 2017](https://arxiv.org/abs/1711.05101))
* `sampling_temperature` parameter to control the randomness of the generation

### Fixes and improvements

* Improve correctness of `MeanEncoder` for variable lengths inputs (requires TensorFlow 1.13+)
* Internal refactoring and changes to prepare for 2.0 transition

## [1.20.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.20.1) (2019-02-22)

### Fixes and improvements

* Fix `score` run type that was broken after some internal refactoring

## [1.20.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.20.0) (2019-02-15)

### New features

* More embeddings sharing combinations:
  * Share embeddings of multi source inputs (set `share_parameters=True` to `ParallelInputter`)
  * Share target embeddings and softmax weights (set `share_embeddings=onmt.models.EmbeddingsSharingLevel.TARGET` to a seq2seq model)
  * Share all embeddings (set `share_embeddings=onmt.models.EmbeddingsSharingLevel.ALL` to a seq2seq model, see the example model in `config/models/transformer_shared_embedding.py`)
* Support converting SentencePiece vocabularies in `onmt-build-vocab`

### Fixes and improvements

* Remove the `--dtype` option of `onmt-ark-to-records`: this is considered a bug fix as the records should always be saved in float32
* Fix output dtype of `SequenceRecordInputter` which was always float32
* Fix guided alignment training for TensorFlow versions older than 1.11
* Refactor the `Inputter` API
* Increase coverage of TensorFlow 2.0 tests and remove temporary namespace `opennmt.v2`

## [1.19.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.19.2) (2019-02-13)

### Fixes and improvements

* Fix error when passing the tokenization configuration as a file in the training configuration

## [1.19.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.19.1) (2019-02-13)

### Fixes and improvements

* Revert default model exporter to "last" because "best" is causing issues for some users

## [1.19.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.19.0) (2019-02-08)

### New features

* Experimental [Horovod](https://github.com/uber/horovod) support
* Experimental `opennmt.v2` namespace that will expose modules compatible with TensorFlow 2.0
* [`sacreBLEU`](https://github.com/mjpost/sacreBLEU) external evaluator (requires Python 3)
* Simplify configuration of non structural arguments: tokenization and pretrained embedding can now be configured in the YAML file directly

### Fixes and improvements

* In distributed training, only the master should save checkpoints
* Clarify loggging of training throughput, use source/target terminology instead of features/labels
* Do not save the learning rate and words per second counters in the checkpoints

## [1.18.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.18.0) (2019-02-01)

### New features

* Argument `--size_multiple` to the `onmt-build-vocab` script to constrain the vocabulary size used during the training
* `TransformerBigFP16` in model catalog

### Fixes and improvements

* Improve FP16 training speed by making the batch size a multiple of 8
* In training logs, dump final run configuration in YAML format instead of JSON

## [1.17.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.17.1) (2019-01-21)

### Fixes and improvements

* Fix offset in alignment vectors introduced by the `<s>` special token
* Fix crash when using the AdafactorOptimizer and setting beta1

## [1.17.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.17.0) (2019-01-10)

### New features

* Experimental batch size autotuning: automatically find the largest supported batch size based on the current configuration and available memory (for token-based batch type only)
* `effective_batch_size` training parameter to automatically configure gradients accumulation based on the current batch size and the number of training replicas
* Add `var_list` argument to the `optimize_loss` function
* `save_checkpoints_secs` training parameter as an alternative to `save_checkpoints_steps`

### Fixes and improvements

* Change default model exporter to ["best"](https://www.tensorflow.org/api_docs/python/tf/estimator/BestExporter) for compatible TensorFlow versions

## [1.16.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.16.0) (2018-12-21)

### New features

* `rnmtplus_decay` learning rate decay function described in [Chen et al. 2018](https://arxiv.org/abs/1804.09849)

### Fixes and improvements

* Fix error when exporting models containing a `SequenceRecordInputter`
* Fix error when using `rsqrt_decay`
* Step logging should respect `save_summary_steps` even when gradients accumulation is used

## [1.15.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.15.0) (2018-11-30)

### New features

* Parameter `sampling_topk` to sample predictions from the output distribution (from [Edunov et al. 2018](https://arxiv.org/abs/1808.09381))

### Fixes and improvements

* Checkpoint utilities now save a relative path instead of absolute in the generated checkpoint state
* Fix error on missing configuration fields that should be optional
* Fix error on gradient accumulation in TensorFlow versions <= 1.9
* Fix optimizer variable names mismatch introduced by gradient accumulation which prevented to continue from an existing checkpoint trained without

## [1.14.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.14.1) (2018-11-28)

### Fixes and improvements

* Fix inference error when using parallel inputs and the parameter `bucket_width`
* Fix size mismatch error when decoding from multi-source models

## [1.14.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.14.0) (2018-11-22)

### New features

* Multi source Transformer architecture with serial attention layers (see the [example model configuration](https://github.com/OpenNMT/OpenNMT-tf/blob/master/config/models/multi_source_transformer.py))
* Inference now accepts the parameter `bucket_width`: if set, the data will be sorted by length to increase the translation efficiency. The predictions will still be outputted in order as they are available. (Enabled by default when using automatic configuration.)

### Fixes and improvements

* Improve greedy decoding speed (up to 50% faster)
* When using `onmt-update-vocab` with the `merge` directive, updated vocabulary files will be saved in the output directory alongside the updated checkpoint

## [1.13.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.13.1) (2018-11-19)

### Fixes and improvements

* Fix error when building an inference graph including a `DenseBridge`

## [1.13.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.13.0) (2018-11-14)

### New features

* RNMT+ decoder
* Parameter `gradients_accum` to accumulate gradients and delay parameters update
* Expose lower-level decoder APIs:
  * `Decoder.step_fn`: returns a callable and an initial state to run step by step decoding
  * `Decoder.decode_from_inputs`: decodes from full inputs (e.g. embeddings)

### Fixes and improvements

* Make learning rate decay configuration more generic: parameters can be set via a `decay_params` map which allows using more meaningful parameters name (see this [example configurations](https://github.com/OpenNMT/OpenNMT-tf/blob/master/config/optim/adam_with_noam_decay.yml))
* By default, auto-configured Transformer models will accumulate gradients to simulate a training with 8 synchronous replicas (e.g. if you train with 4 GPUs, the gradients of 2 consecutive steps will be accumulated)

## [1.12.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.12.0) (2018-11-07)

### New features

* The command line argument `--checkpoint_path` can be used to load the weights of an existing checkpoint while starting from a fresh training state (i.e. with new learning rate schedule and optimizer variables)
* Parameter `minimum_decoding_length` to constrain the minimum length of decoded sequences

### Fixes and improvements

* Major refactoring of dynamic decoding internals: decoding loops are now shared between all decoders that should only implement a step function
* Move event files of external evaluators to the `eval/` subdirectory
* Report non normalized hypotheses score for clarity

## [1.11.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.11.0) (2018-10-24)

### New features

* `onmt-convert-checkpoint` script to convert checkpoints from one data type to another (e.g. train with FP16 but export in FP32)
* Additional output options for the `score` run type:
  * `with_token_level` to output the score of each token
  * `with_alignments` to output the source-target alignments
* Display the package version by running `onmt-main -v`

### Fixes and improvements

* Fix error in `SelfAttentionDecoder` when `memory` is not defined (e.g. in LM tasks)
* Fix `UnicodeDecodeError` when printing predictions on the standard output in Docker containers
* Force `onmt-update-vocab` script to run on CPU
* Raise error if distributed training is configured but the `train_and_eval` run type is not used

## [1.10.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.10.1) (2018-10-15)

### Fixes and improvements

* Fix possible error when loading checkpoints without `--model_type` or `--model` after updating to a newer OpenNMT-tf version. The saved model description is now more future-proof regarding model class updates.
* Fix embedding visualization when the vocabulary file is stored in the model directory or when a joint vocabulary is used
* Improve encoder/decoder states compatibility check

## [1.10.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.10.0) (2018-10-11)

### New features

* `--auto_config` flag to use automatic training configuration values (e.g. optimizer, learning rate, batch size). For compatible models, the automatic values aim to deliver solid performance out of the box.
* Include all tokenization assets in exported models

### Fixes and improvements

* Fix type error during evaluation and inference of FP16 Transformer models
* Update the [model serving example](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) to use a real pretrained model with the TensorFlow Serving 1.11 GPU Docker image
* Small training speed improvement when the optimizer implements sparse updates
* Revise some default configuration values:
  * change `bucket_width` default value to 1 (from 5)
  * change inference `batch_size` default value to 16 (from 1)

## [1.9.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.9.0) (2018-10-05)

### New features

* Mixed precision training of Transformer models
* Command line option `--export_dir_base` to configure the destination directory of manually exported models

### Fixes and improvements

* Fix error when loading model configuration containing the `OpenNMTTokenizer` tokenizer
* Include `OpenNMTTokenizer` subword models in the graph assets

## [1.8.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.8.1) (2018-09-28)

### Fixes and improvements

* Fix backward incompatible change made to `Model.__call__` output types

## [1.8.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.8.0) (2018-09-25)

### New features

* Guided alignment for models using `SelfAttentionDecoder` and `AttentionalRNNDecoder`
* `with_scores` inference option to also output the prediction score
* `with_alignments` inference option to also output the source-target alignments

### Fixes and improvements

* `SelfAttentionDecoder` defines the first attention head of the last layer as its source-target attention vector

## [1.7.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.7.0) (2018-08-07)

### New features

* Command line option `--session_config` to configure TensorFlow session parameters (see the "Configuration" documentation)
* `share_embeddings` argument to `SequenceToSequence` models to configure the level of embeddings sharing

### Fixes and improvements

* Fix error when using `--data_dir` and parallel inputs in the data configuration
* Fix TensorFlow 1.9+ compatibility issue when using `MultiplyReducer`
* Better support of other filesystems (HDFS, S3, etc.) for the model directory and data files

## [1.6.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.6.2) (2018-07-14)

### Fixes and improvements

* Fix invalid scheduled sampling implementation with RNN decoders
* Fix possible "Data loss: Invalid size in bundle entry" error when loading an averaged checkpoint
* Fix `PyramidalEncoder` error when input lengths are smaller than the total reduction factor

## [1.6.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.6.1) (2018-07-11)

### Fixes and improvements

* Fix error when initialiazing the ROUGE evaluator
* Improve Transformer models performance:
  * fix performance regression of `tf.layers.conv1d` on TensorFlow 1.7+ (+20%)
  * better caching during decoding (+15%)

## [1.6.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.6.0) (2018-07-05)

### New features

* New model exporter types:
  * `best` to export a new model only if it achieves the best evaluation loss so far (requires TensorFlow 1.9+)
  * `final` to only export the model at the end of the training
* Script and API to map a checkpoint to new vocabularies while keeping the trained weights of common words

### Fixes and improvements

* Fix error when reloading models with target pretrained embeddings
* Fix error message when the number of requested GPUs is incompatible with the number of visible devices
* Fix error when continuing the training from an averaged checkpoint
* Re-introduce `rouge` as a dependency since its installation is now fixed
* Make code forward compatible with future `pyonmttok` options

## [1.5.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.5.0) (2018-06-08)

### New features

* `MultistepAdamOptimizer` to simulate trainings with large batch size (credits to Tensor2Tensor, requires TensorFlow 1.6+)
* `--log_prediction_time` flag to summarize inference execution time:
  * total prediction time
  * average prediction time
  * tokens per second
* Training option `average_last_checkpoints` to automatically average checkpoints at the end of the training
* Command line options `--{inter,intra}_op_parallelism_threads` to control the level of CPU parallelism
* [*experimental*] Average attention network ([Zhang et al. 2018](https://arxiv.org/abs/1805.00631)) in the Transformer decoder

### Fixes and improvements

* Fix possible error when training RNN models with in-graph replication (time dimension mismatch)
* Fix error when the set vocabulary file is in the model directory

## [1.4.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.4.1) (2018-05-25)

### Fixes and improvements

* Make `rouge` an optional dependency to avoid install error in a fresh environment

## [1.4.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.4.0) (2018-05-25)

### New features

* `score` run type to score existing predictions
* ROUGE external evaluator for summarization
* `CharRNNEmbedder` that runs a RNN layer over character embeddings
* High level APIs for efficient data pipelines (see `utils.data.{training,inference}_pipeline`)

### Fixes and improvements

* Add more control over model export after evaluation with the `exporters` option
* Allow `JoinReducer` to be used on the `ParallelEncoder` output

## [1.3.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.3.0) (2018-05-14)

### New features

* RNMT+ encoder
* L1, L2, and L1 L2 regularization penalties (see `regularization` parameter)
* Support additional post processing layers in `ParallelEncoder`:
  * `outputs_layer_fn` applied on each encoder outputs
  * `combined_output_layer_fn` applied on the combined output

### Fixes and improvements

* Fix `SequenceClassifier` "last" encoding for variable length sequences

## [1.2.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.2.0) (2018-04-28)

### New features

* Return alignment history when decoding from an `AttentionalRNNDecoder` (requires TensorFlow 1.8+ when decoding with beam search)
* Boolean parameter `replace_unknown_target` to replace unknown target tokens by the source token with the highest attention (requires a decoder that returns the alignment history)
* Support for arbitrary transition layers in `SequentialEncoder`

### Fixes and improvements

* Fix sequence reduction when the maximum sequence length is not equal to the tensor time dimension (e.g. when splitting a batch for multi-GPU training)
* The number of prefetched batches is automatically tuned when `prefetch_buffer_size` is not set (for TensorFlow 1.8+)

## [1.1.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.1.0) (2018-04-12)

### New features

* Update the OpenNMT tokenizer to 1.3.0 and use its Python package instead of requiring a manual compilation (Linux only)
* Include a catalog of models in the library package and allow model selection with the `--model_type` command line option

### Fixes and improvements

* Fix error when using FP16 and an `AttentionMechanism` module (for TensorFlow 1.5+)
* Manual export will remove default-valued attributes from the NodeDefs (for TensorFlow 1.6+)
* Silence some deprecation warnings with recent TensorFlow versions
* Training option `sample_buffer_size` now accepts special values:
  * `0` or `null` to disable shuffling
  * `-1` to create a buffer with the same size as the training dataset

## [1.0.3](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.0.3) (2018-04-02)

### Fixes and improvements

* Make `Runner.export` return the path to the export directory
* Fix and update `setup.py` to support `pip` installation

## [1.0.2](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.0.2) (2018-03-28)

### Fixes and improvements

* Fix the encoder state structure when RNN encoders are combined (e.g. in `SequentialEncoder`)
* Fix `CharConvEmbedder` error on empty sequences
* Fix `Adafactor` crash on sparse updates, automatically fallback to dense updates instead
* Improve the Transformer decoder mask construction (up to 10% speedup during training)

## [1.0.1](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.0.1) (2018-03-14)

### Fixes and improvements

* Fix undefined `xrange` error in `utils/beam_search.py` when using Python 3

## [1.0.0](https://github.com/OpenNMT/OpenNMT-tf/releases/tag/v1.0.0) (2018-03-14)

Initial stable release.
