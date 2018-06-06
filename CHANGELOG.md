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

* `MultistepAdamOptimizer` to simulate trainings with large batch size (credits to Tensor2Tensor, requires TensorFlow 1.6+)
* `--log_prediction_time` flag to summarize inference execution time:
  * total prediction time
  * average prediction time
  * tokens per second
* Training option `average_last_checkpoints` to automatically average checkpoints at the end of the training
* [*experimental*] Average attention network ([Zhang et al. 2018](https://arxiv.org/abs/1805.00631)) in the Transformer decoder

### Fixes and improvements

* Fix possible error when training RNN models with in-graph replication (time dimension mismatch)

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
