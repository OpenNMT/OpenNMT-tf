[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT-tf.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-tf) [![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](http://opennmt.net/OpenNMT-tf/) [![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-tf.svg)](https://gitter.im/OpenNMT/OpenNMT-tf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence modeling tool in TensorFlow with production in mind. While neural machine translation is the main target task, it has been designed to more generally support:

* sequence to sequence mapping
* sequence tagging
* sequence classification

*The project is still in beta and does not come with stability guarantees.*

## Key features

OpenNMT-tf focuses on modularity to support advanced modeling and training capabilities:

* **arbitrarily complex encoder architectures**<br/>e.g. mixing RNNs, CNNs, self-attention, etc. in parallel or in sequence.
* **hybrid encoder-decoder models**<br/>e.g. self-attention encoder and RNN decoder or vice versa.
* **multi-source training**<br/>e.g. source text and Moses translation as inputs for machine translation.
* **multiple input format**<br/>text with support of mixed word/character embeddings or real vectors serialized in *TFRecord* files.
* **on-the-fly tokenization**<br/>apply advanced tokenization dynamically during the training and detokenize the predictions during inference or evaluation.
* **automatic evaluation**<br/>support for saving evaluation predictions and running external evaluators (e.g. BLEU).

and all of the above can be used simultaneously to train novel and complex architectures. See the [predefined models](config/models) to discover how they are defined.

OpenNMT-tf is also compatible with some of the best TensorFlow features:

* asynchronous distributed training
* monitoring with [TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
* inference with [TensorFlow Serving](https://www.tensorflow.org/serving/) and the TensorFlow C++ API.

## Requirements

* `tensorflow` (`1.4`)
* `pyyaml`

## Overview

A minimal OpenNMT-tf run consists of 3 elements:

* a Python file describing the **model**
* a YAML file describing the **parameters**
* a **run** type

e.g.:

```
python -m bin.main <run_type> --model <model_file.py> --config <config_file.yml>
```

When loading an existing checkpoint, the `--model` option is optional.

* For more information about configuration files, see the [documentation](http://opennmt.net/OpenNMT-tf/configuration.html).
* For more information about command line options, see the help flag `python -m bin.main -h`.

## Quickstart

Here is a minimal workflow to get you started in using OpenNMT-tf. This example uses a toy English-German dataset for machine translation.

1\. Build the word vocabularies:

```
python -m bin.build_vocab --size 50000 --save_vocab data/toy-ende/src-vocab.txt data/toy-ende/src-train.txt
python -m bin.build_vocab --size 50000 --save_vocab data/toy-ende/tgt-vocab.txt data/toy-ende/tgt-train.txt
```

2\. Train with preset parameters:

```
python -m bin.main train --model config/models/nmt_small.py --config config/opennmt-defaults.yml config/data/toy-ende.yml
```

3\. Translate a test file with the latest checkpoint:

```
python -m bin.main infer --config config/opennmt-defaults.yml config/data/toy-ende.yml --features_file data/toy-ende/src-test.txt
```

**Note:** do not expect any good translation results with this toy example. Consider training on [larger parallel datasets](http://www.statmt.org/wmt16/translation-task.html) instead.

For more advanced usages, see the [documentation](http://opennmt.net/OpenNMT-tf).

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
