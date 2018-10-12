# Training

## Monitoring

OpenNMT-tf uses [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard) to log information during the training. Simply start `tensorboard` by setting the active log directory, e.g.:

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

To enable distributed training, the user should use the `train_and_eval` run type and set on the command line:

* a **chief worker** host that runs a training loop and manages checkpoints, summaries, etc.
* a list of **worker** hosts that run a training loop
* a list of **parameter server** hosts that synchronize the parameters

Then a training instance should be started on each host with a selected task, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 onmt-main train_and_eval [...] \
    --ps_hosts localhost:2222 \
    --chief_host localhost:2223 \
    --worker_hosts localhost:2224,localhost:2225 \
    --task_type worker \
    --task_index 1
```

will start the worker 1 on the current machine and first GPU. By setting `CUDA_VISIBLE_DEVICES` correctly, asynchronous distributed training can be run on a single multi-GPU machine.

For more details, see the documentation of [`tf.estimator.train_and_evaluate`](https://www.tensorflow.org/api_docs/python/tf/estimator/train_and_evaluate). Also see [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) to integrate distributed training with open-source frameworks like Docker or Kubernetes.

**Note:** distributed training will also split the training directory `model_dir` accross the instances. This could impact features that restore checkpoints like inference, manual export, or checkpoint averaging. The recommend approach to properly support these features while running distributed training is to store the `model_dir` on a shared filesystem, e.g. by using [HDFS](https://www.tensorflow.org/deploy/hadoop).

## Mixed precision training

Thanks to [work from NVIDIA](https://github.com/NVIDIA/OpenSeq2Seq), OpenNMT-tf supports training models using FP16 computation. Mixed precision training is automatically enabled when the data type of the [inputters](package/opennmt.inputters.inputter.html) is defined to be `tf.float16`. See for example the predefined model `TransformerFP16`:

```bash
onmt-main train [...] --model_type TransformerFP16
```

Additional training configurations are available to tune the loss scaling algorithm:

```yaml
params:
  # (optional) For mixed precision training, the loss scaling to apply (a constant value or
  # an automatic scaling algorithm: "backoff", "logmax", default: "backoff")
  loss_scale: backoff
  # (optional) For mixed precision training, the additional parameters to pass the loss scale
  # (see the source file opennmt/optimizers/mixed_precision_wrapper.py).
  loss_scale_params:
    scale_min: 1.0
    step_factor: 2.0
```

For more information about the implementation and get expert recommendation on how to maximize performance, see the [OpenSeq2Seq's documentation](https://nvidia.github.io/OpenSeq2Seq/html/mixed-precision.html). Currently, mixed precision training requires Volta GPUs and the NVIDIA's TensorFlow Docker image.
