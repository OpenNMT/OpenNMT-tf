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

* `onmt-convert-checkpoint` script to convert checkpoints from one data type to another (e.g. train with FP16 but export in FP32)
* Additional output options for the `score` run type:
  * `with_token_level` to output the score of each token
  * `with_alignments` to output the source-target alignments

### Fixes and improvements

* Fix error in `SelfAttentionDecoder` when `memory` is not defined (e.g. in LM tasks)
* Fix `UnicodeDecodeError` when printing predictions on the standard output in Docker containers

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
