[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT-tf.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-tf) [![PyPI version](https://badge.fury.io/py/OpenNMT-tf.svg)](https://badge.fury.io/py/OpenNMT-tf) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](http://opennmt.net/OpenNMT-tf/) [![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-tf.svg)](https://gitter.im/OpenNMT/OpenNMT-tf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence learning toolkit using TensorFlow. While neural machine translation is the main target task, it has been designed to more generally support:

* sequence to sequence mapping
* sequence tagging
* sequence classification
* language modeling

The project is production-oriented and comes with [backward compatibility guarantees](CHANGELOG.md).

## Key features

OpenNMT-tf focuses on modularity to support advanced modeling and training capabilities:

* **arbitrarily complex encoder architectures**<br/>e.g. mixing RNNs, CNNs, self-attention, etc. in parallel or in sequence.
* **hybrid encoder-decoder models**<br/>e.g. self-attention encoder and RNN decoder or vice versa.
* **neural source-target alignment**<br/>train with guided alignment to constrain attention vectors and output alignments as part of the translation API.
* **multi-source training**<br/>e.g. source text and Moses translation as inputs for machine translation.
* **multiple input format**<br/>text with support of mixed word/character embeddings or real vectors serialized in *TFRecord* files.
* **on-the-fly tokenization**<br/>apply advanced tokenization dynamically during the training and detokenize the predictions during inference or evaluation.
* **domain adaptation**<br/>specialize a model to a new domain in a few training steps by updating the word vocabularies in checkpoints.
* **automatic evaluation**<br/>support for saving evaluation predictions and running external evaluators (e.g. BLEU).
* **mixed precision training**<br/>take advantage of the latest NVIDIA optimizations to train models with half-precision floating points.

and all of the above can be used simultaneously to train novel and complex architectures. See the [predefined models](opennmt/models/catalog.py) to discover how they are defined and the [API documentation](http://opennmt.net/OpenNMT-tf/package/opennmt.html) to customize them.

OpenNMT-tf is also compatible with some of the best TensorFlow features:

* multi-GPU training
* distributed training via [Horovod](https://github.com/uber/horovod) or TensorFlow asynchronous training
* monitoring with [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
* inference with [TensorFlow Serving](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) and the [TensorFlow C++ API](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/cpp)

## Usage

OpenNMT-tf requires:

* Python >= 2.7
* TensorFlow >= 1.4, < 2.0

We recommend installing it with `pip`:

```bash
pip install OpenNMT-tf
```

*See the [documentation](http://opennmt.net/OpenNMT-tf/installation.html) for more information.*

### Command line

OpenNMT-tf comes with several command line utilities to prepare data, train, and evaluate models.

For all tasks involving a model execution, OpenNMT-tf uses a unique entrypoint: `onmt-main`. A typical OpenNMT-tf run consists of 3 elements:

* the **run** type: `train_and_eval`, `train`, `eval`, `infer`, `export`, or `score`
* the **model** type
* the **parameters** described in a YAML file

that are passed to the main script:

```
onmt-main <run_type> --model_type <model> --auto_config --config <config_file.yml>
```

*For more information and examples on how to use OpenNMT-tf, please visit [our documentation](http://opennmt.net/OpenNMT-tf).*

### Library

OpenNMT-tf also exposes well-defined and stable APIs. Here is an example using the library to encode a sequence using a self-attentional encoder:

```python
import tensorflow as tf
import opennmt as onmt

tf.enable_eager_execution()

# Build a random batch of input sequences.
inputs = tf.random.uniform([3, 6, 256])
sequence_length = tf.constant([4, 6, 5], dtype=tf.int32)

# Encode with a self-attentional encoder.
encoder = onmt.encoders.SelfAttentionEncoder(num_layers=6)
outputs, _, _ = encoder.encode(
    inputs,
    sequence_length=sequence_length,
    mode=tf.estimator.ModeKeys.TRAIN)

print(outputs)
```

For more advanced examples, some online resources are using OpenNMT-tf as a library:

* The directory `examples/library` contains additional examples that use OpenNMT-tf as a library
* [OpenNMT Hackathon 2018](https://github.com/OpenNMT/Hackathon/tree/master/unsupervised-nmt) features a tutorial to implement unsupervised NMT using OpenNMT-tf
* [nmt-wizard-docker](https://github.com/OpenNMT/nmt-wizard-docker) uses the high-level `onmt.Runner` API to wrap OpenNMT-tf with a custom interface for training, translating, and serving

*For a complete overview of the APIs, see the [package documentation](http://opennmt.net/OpenNMT-tf/package/opennmt.html).*

## Compatibility with {Lua,Py}Torch implementations

OpenNMT-tf has been designed from scratch and compatibility with the {Lua,Py}Torch implementations in terms of usage, design, and features is not a priority. Please submit a feature request for any missing feature or behavior that you found useful in the {Lua,Py}Torch implementations.

## Acknowledgments

The implementation is inspired by the following:

* [TensorFlow's NMT tutorial](https://github.com/tensorflow/nmt)
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Google's seq2seq](https://github.com/google/seq2seq)
* [OpenSeq2Seq](https://github.com/NVIDIA/OpenSeq2Seq)

## Additional resources

* [Documentation](http://opennmt.net/OpenNMT-tf)
* [Forum](http://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/OpenNMT-tf)
