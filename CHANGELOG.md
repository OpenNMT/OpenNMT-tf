**Notes on versioning**

OpenNMT-tf follows [semantic versioning 2.0.0](https://semver.org/). The API covers:

* command line options
* configuration files
* checkpoints
* public classes and functions that do not come from third parties
* minimum required TensorFlow version

---

## [Unreleased]

### Breaking changes

### New features

* Update the OpenNMT tokenizer to 1.3.0 and use its Python package instead of requiring a manual compilation (Linux only)

### Fixes and improvements

* Fix error when using FP16 and an `AttentionMechanism` module (for TensorFlow 1.5+)
* Manual export will remove default-valued attributes from the NodeDefs (for TensorFlow 1.6+)
* Silence some deprecation warnings with recent TensorFlow versions

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
