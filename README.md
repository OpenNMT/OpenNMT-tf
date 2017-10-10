*This project is experimental. Feedback and testing are welcome.*

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence modeling tool in TensorFlow for:

* sequence to sequence mapping
* sequence tagging
* sequence classification

It focuses on modularity and extensibility using standard TensorFlow modules and practices.

## Requirements

* `tensorflow`
  * >= [`4da840dd`](https://github.com/tensorflow/tensorflow/commit/4da840dd946da3d253ee88e1757ba8a0c0f130f0) to use `crf_decoding=True`
  * >= [`865b92da`](https://github.com/tensorflow/tensorflow/commit/865b92da01582081576728504bedf932b367b26c) to use `CharConvEmbedder`
  * >= `1.3.0` otherwise
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

### Model configuration

Models are defined from the code to allow a high level of modeling freedom. The user should provide a `opennmt.models.Model` instance using available or user-defined modules.

Some modules are defined to contain other modules and can be used to design complex architectures:

* `opennmt.encoders.ParallelEncoder`
* `opennmt.encoders.SequentialEncoder`
* `opennmt.inputters.MixedInputter`
* `opennmt.inputters.ParallelInputter`

*See the template file `config/models/template.py` and examples in `config/models/`.*

### Parameters configuration

Parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference.

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```
python -m bin.main [...] --config config/data/wmt_ende.yml config/run/default_train.yml config/params/adam_with_decay.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

*See the example configuration `config/config.sample.yml` to learn about available options.*

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

## Distributed training

OpenNMT-tf provides asynchronous distributed training (see the [TensorFlow documentation](https://www.tensorflow.org/deploy/distributed) for a general description of distributed training). The user should provide on the command line:

* a list of **workers** hosts that run a training loop; the first worker is the master and also manages checkpoints, summaries, and evaluation
* a list of **parameter server** hosts that synchronize the parameters

Then a training instance should be started on each host with a selected task, e.g.:

```
CUDA_VISIBLE_DEVICES=0 python -m bin.main train [...] --ps_hosts localhost:2222 --worker_hosts localhost:2223,localhost:2224 --task_type worker --task_index 1
```

will start the worker 1 on the current machine and first GPU.

Also see [`tensorflow/ecosystem`](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.

## Model serving

OpenNMT-tf can export models for inference in other environments, for example with [TensorFlow Serving](https://www.tensorflow.org/serving/). A model export contains all information required for inference: the graph definition, the weights, and external assets such as vocabulary files. It typically looks like this on disk:

```
models/enfr/1507109306/
├── assets
│   ├── en.dict
│   └── fr.dict
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

Models are automatically exported at the end of the training or manually with the `export` run type.

When using an exported model, you should prepare the inputs as expected by the model. See the `_get_serving_input` methods of the inputters modules of your model to define the data that need to be fed. Some examples are also available in the `examples/` directory:

* `examples/serving` to serve a model with TensorFlow Serving
* `examples/cpp` to run inference with the TensorFlow C++ API

**Note:** because the Python function used in `tf.py_func` is not serialized in the graph, model exports do not support in-graph tokenization and text inputs are expected to be tokenized.
