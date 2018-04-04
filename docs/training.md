# Training

## Monitoring

OpenNMT-tf uses [TensorBoard](https://github.com/tensorflow/tensorboard) to log information during the training. Simply start `tensorboard` by setting the active log directory, e.g.:

```bash
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

## Replicated training

OpenNMT-tf training can make use of multiple GPUs with *in-graph replication*. In this mode, the main section of the graph is replicated over multiple devices and batches are processed in parallel. The resulting graph is equivalent to train with batches `N` times larger, where `N` is the number of used GPUs.

For example, if your machine has 4 GPUs, simply add the `--num_gpus` option:

```bash
onmt-main train [...] --num_gpus 4
```

Note that evaluation and inference will run on a single device.

## Distributed training

OpenNMT-tf also supports asynchronous distributed training with *between-graph replication*. In this mode, each graph replica processes a batch independently, compute the gradients, and asynchronously update a shared set of parameters.

To enable distributed training, the user should set on the command line:

* a **chief worker** host that runs a training loop and manages checkpoints, summaries, etc.
* a list of **worker** hosts that run a training loop
* a list of **parameter server** hosts that synchronize the parameters

Then a training instance should be started on each host with a selected task, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 onmt-main train [...] --ps_hosts localhost:2222 --chief_host localhost:2223 --worker_hosts localhost:2224,localhost:2225 --task_type worker --task_index 1
```

will start the worker 1 on the current machine and first GPU. By setting `CUDA_VISIBLE_DEVICES` correctly, asynchronous distributed training can be run on a single multi-GPU machine.

For more details, see the documentation of [`tf.estimator.train_and_evaluate`](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate). Also see [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.
