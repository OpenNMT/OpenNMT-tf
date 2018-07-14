[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT-tf.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-tf) [![PyPI version](https://badge.fury.io/py/OpenNMT-tf.svg)](https://badge.fury.io/py/OpenNMT-tf) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](http://opennmt.net/OpenNMT-tf/) [![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-tf.svg)](https://gitter.im/OpenNMT/OpenNMT-tf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence modeling tool in TensorFlow with production in mind. While neural machine translation is the main target task, it has been designed to more generally support:

* sequence to sequence mapping
* sequence tagging
* sequence classification

## Key features

OpenNMT-tf focuses on modularity to support advanced modeling and training capabilities:

* **arbitrarily complex encoder architectures**<br/>e.g. mixing RNNs, CNNs, self-attention, etc. in parallel or in sequence.
* **hybrid encoder-decoder models**<br/>e.g. self-attention encoder and RNN decoder or vice versa.
* **multi-source training**<br/>e.g. source text and Moses translation as inputs for machine translation.
* **multiple input format**<br/>text with support of mixed word/character embeddings or real vectors serialized in *TFRecord* files.
* **on-the-fly tokenization**<br/>apply advanced tokenization dynamically during the training and detokenize the predictions during inference or evaluation.
* **automatic evaluation**<br/>support for saving evaluation predictions and running external evaluators (e.g. BLEU).

and all of the above can be used simultaneously to train novel and complex architectures. See the [predefined models](opennmt/models/catalog.py) to discover how they are defined and the [API documentation](http://opennmt.net/OpenNMT-tf/package/opennmt.html) to customize them.

OpenNMT-tf is also compatible with some of the best TensorFlow features:

* replicated and distributed training
* monitoring with [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
* inference with [TensorFlow Serving](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) and the [TensorFlow C++ API](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/cpp)

## Requirements

* Python (>= 2.7)
* TensorFlow (>= 1.4)

*(Some features may require newer TensorFlow versions, see the [changelog](CHANGELOG.md).)*

## Installation

```bash
pip install OpenNMT-tf
```

## Overview

A minimal OpenNMT-tf run consists of 3 elements:

* the **run** type: `train_and_eval`, `train`, `eval`, `infer`, `export`, or `score`
* the **model** type
* the **parameters** described in a YAML file

that are passed to the main script:

```
onmt-main <run_type> --model_type <model> --config <config_file.yml>
```

Additional experimental models are available in the `config/models/` directory and can be used with the option `--model <model_file.py>`.

* For more information about configuration files, see the [documentation](http://opennmt.net/OpenNMT-tf/configuration.html).
* For more information about command line options, see the help flag `onmt-main -h`.

## Quickstart

Here is a minimal workflow to get you started in using OpenNMT-tf. This example uses a toy English-German dataset for machine translation.

1\. Clone the repository to fetch the sample data and the predefined configurations:

```
git clone --depth 1 --branch r1 --single-branch https://github.com/OpenNMT/OpenNMT-tf.git
cd OpenNMT-tf
```

2\. Build the word vocabularies:

```
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/src-vocab.txt data/toy-ende/src-train.txt
onmt-build-vocab --size 50000 --save_vocab data/toy-ende/tgt-vocab.txt data/toy-ende/tgt-train.txt
```

3\. Train with preset parameters:

```
onmt-main train_and_eval --model_type NMTSmall --config config/opennmt-defaults.yml config/data/toy-ende.yml
```

4\. Translate a test file with the latest checkpoint:

```
onmt-main infer --config config/opennmt-defaults.yml config/data/toy-ende.yml --features_file data/toy-ende/src-test.txt
```

**Note:** do not expect any good translation results with this toy example. Consider training on [larger parallel datasets](http://www.statmt.org/wmt16/translation-task.html) instead.

*For more advanced usages, see the [documentation](http://opennmt.net/OpenNMT-tf) or the [WMT training scripts](https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts/wmt).*

## Using as a library

OpenNMT-tf also exposes well-defined and stable APIs. Here is an example using the library to encode a sequence using a self-attentional encoder:

```python
import tensorflow as tf
import opennmt as onmt

# Build a random batch of input sequences.
sequence_length = [4, 6, 5]
input_depth = 512
inputs = tf.placeholder_with_default(
    np.random.randn(
        len(sequence_length), max(sequence_length), input_depth).astype(np.float32),
    shape=(None, None, input_depth))

# Encode with a self-attentional encoder.
encoder = onmt.encoders.SelfAttentionEncoder(num_layers=4)
outputs, state, outputs_length = encoder.encode(
    inputs,
    sequence_length=sequence_length,
    mode=tf.estimator.ModeKeys.TRAIN)
```

For more advanced examples, some online resources are using OpenNMT-tf as a library:

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

## Additional resources

* [Documentation](http://opennmt.net/OpenNMT-tf)
* [Forum](http://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/OpenNMT-tf)
