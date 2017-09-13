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

An OpenNMT-tf run consists of two elements:

* a Python file describing the **model**
* a YAML file describing the **run**

e.g.:

```
python onmt.py --model config/models/my_model.py --run config/my_run.yml
```

### Model configuration

Models are defined from the code to allow a high level of modeling freedom and ensuring that the library exposes consistent interfaces. The user should provide a `opennmt.models.Model` instance using available or user-defined modules.

*See the template file `config/models/template.py` and examples in `config/models/`.*

### Run configuration

Runs are described in separate YAML files. They define how to train, infer or export a model.

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```
python onmt.py [...] --run config/data/wmt_ende.yml config/run/default_train.yml config/params/adam_with_decay.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority. Additionally, fields marked as optional have a default value defined in `opennmt/config.py`.

*See example configurations in `config/`.*

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
CUDA_VISIBLE_DEVICES=0 python onmt.py [...] --ps_hosts localhost:2222 --worker_hosts localhost:2223,localhost:2224 --task_type worker --task_index 1
```

will start the worker 1 on the current machine and first GPU.

Also see [`tensorflow/ecosystem`](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.
