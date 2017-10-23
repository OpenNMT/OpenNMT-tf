*This project is experimental. Feedback and testing are welcome.*

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence modeling tool in TensorFlow for:

* sequence to sequence mapping
* sequence tagging
* sequence classification

It focuses on modularity and extensibility using standard TensorFlow modules and practices.

## Requirements

* `tensorflow` (`1.4.0rc0`)
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

When loading an existing checkpoint, the `--model` option is optional. See the help flag `python -m bin.main -h` to discover additional command line options.

### Model configuration

Models are defined from the code to allow a high level of modeling freedom. The user should provide a `opennmt.models.Model` instance using available or user-defined modules.

Some modules are defined to contain other modules and can be used to design complex architectures:

* `opennmt.encoders.ParallelEncoder`
* `opennmt.encoders.SequentialEncoder`
* `opennmt.inputters.MixedInputter`
* `opennmt.inputters.ParallelInputter`

For example, these container modules can be used to implement multi source inputs, multi modal training, mixed word/character embeddings, and arbitrarily complex encoder architectures (e.g. mixing convolution, RNN, self-attention, etc.).

*See the template file `config/models/template.py` and predefined models in `config/models/`. Contributions to add more model configurations are welcome.*

### Parameters configuration

Parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference.

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```
python -m bin.main [...] --config config/data/wmt_ende.yml config/run/default_train.yml config/params/adam_with_decay.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

*See the example configuration `config/sample.yml` to learn about available parameters.*

## Quickstart

Here is a minimal workflow to get you started in using OpenNMT-tf. This example uses a toy English-German dataset for machine translation.

1\. Build the word vocabularies:

```
python -m bin.build_vocab --save_vocab data/toy-ende/src-vocab.txt data/toy-ende/src-train.txt
python -m bin.build_vocab --save_vocab data/toy-ende/tgt-vocab.txt data/toy-ende/tgt-train.txt
```

2\. Train with preset parameters:

```
python -m bin.main train --model config/models/nmt_medium.py --config config/opennmt-defaults.yml config/data/toy-ende.yml
```

4\. Translate a test file with the latest checkpoint:

```
python -m bin.main infer --config config/opennmt-defaults.yml config/data/toy-ende.yml --features_file data/toy-ende/src-test.txt
```

**Note:** do not expect any good translation results with this toy example. Consider training on [larger parallel datasets](http://www.statmt.org/wmt16/translation-task.html) instead.

## Monitoring

OpenNMT-tf uses [TensorBoard](https://github.com/tensorflow/tensorboard) to log information during the training. Simply start `tensorboard` by setting the active log directory, e.g.:

```
tensorboard --logdir="."
```

then open the URL displayed in the shell to monitor and visualize several data, including:

* training and evaluation loss
* training speed
* learning rate
* gradients norm
* computation graphs
* word embeddings
* decoder sampling probability

## Distributed training

OpenNMT-tf supports asynchronous distributed training. The user should set on the command line:

* a **chief worker** host that runs a training loop and manages checkpoints, summaries, etc.
* a list of **worker** hosts that run a training loop
* a list of **parameter server** hosts that synchronize the parameters

Then a training instance should be started on each host with a selected task, e.g.:

```
CUDA_VISIBLE_DEVICES=0 python -m bin.main train [...] --ps_hosts localhost:2222 --chief_host localhost:2223 --worker_hosts localhost:2224,localhost:2225 --task_type worker --task_index 1
```

will start the worker 1 on the current machine and first GPU.

For more details, see the documentation of [`tf.estimator.train_and_evaluate`](https://www.tensorflow.org/versions/r1.4/api_docs/python/tf/estimator/train_and_evaluate). Also see [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.

## Model serving

OpenNMT-tf periodically exports models for inference in other environments, for example with [TensorFlow Serving](https://www.tensorflow.org/serving/). A model export contains all information required for inference: the graph definition, the weights, and external assets such as vocabulary files. It typically looks like this on disk:

```
toy-ende/export/latest/1507109306/
├── assets
│   ├── src-vocab.txt
│   └── tgt-vocab.txt
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

Models are automatically exported during the training or manually with the `export` run type.

When using an exported model, you need to know the input and output nodes of your model. You can use the [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) script provided by TensorFlow for inspection, e.g.:

```
saved_model_cli show --dir toy-ende/export/latest/1507109306/ --tag_set serve --signature_def serving_default
```

Some examples using exported models are available in the `examples/` directory:

* `examples/serving` to serve a model with TensorFlow Serving
* `examples/cpp` to run inference with the TensorFlow C++ API

**Note:** because the Python function used in `tf.py_func` is not serialized in the graph, model exports do not support in-graph tokenization and text inputs are expected to be tokenized.
