**Notes on versioning**

OpenNMT-tf follows [semantic versioning 2.0.0](https://semver.org/). The API covers:

* command line options
* configuration files
* checkpoints of non experimental models
* public classes and functions that do not come from third parties
* minimum required TensorFlow version

---

## [Unreleased]

### New features

### Fixes and improvements

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
