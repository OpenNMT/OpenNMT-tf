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

### via TensorFlow asynchronous training

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

### via Horovod (experimental)

OpenNMT-tf has an experimental support for [Horovod](https://github.com/uber/horovod), enabled when the flag `--horovod` is passed to the command line. For example, this command starts a training on 4 GPUs (don't use Horovod in that case, just use `--num_gpus`):

```bash
mpirun -np 4 \
    -H localhost:4 \
    -bind-to none -map-by slot \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    onmt-main train --model_type Transformer --config data.yml --auto_config --horovod
```

Additional parameters can be set in the training configuration:

```yaml
params:
  # (optional) Horovod parameters.
  horovod:
    # (optional) Compression type for gradients (can be: "none", "fp16", default: "none").
    compression: none
    # (optional) Average the reduced gradients (default: false).
    average_gradients: false
```

For more information on how to install and use Horovod, please see the [GitHub repository](https://github.com/uber/horovod).

**Note:** distributed training will also split the training directory `model_dir` accross the instances. This could impact features that restore checkpoints like inference, manual export, or checkpoint averaging. The recommend approach to properly support these features while running distributed training is to store the `model_dir` on a shared filesystem, e.g. by using [HDFS](https://www.tensorflow.org/deploy/hadoop).

## Mixed precision training

Thanks to [work from NVIDIA](https://github.com/NVIDIA/OpenSeq2Seq), OpenNMT-tf supports training models using FP16 computation. Mixed precision training is automatically enabled when the data type of the [inputters](package/opennmt.inputters.inputter.html) is defined to be `tf.float16`. See for example the predefined model `TransformerFP16`, which is up to [1.8x faster](https://github.com/OpenNMT/OpenNMT-tf/pull/211#issuecomment-455605090) than the FP32 version on compatible hardware:

```bash
onmt-main train_and_eval --model_type TransformerFP16 --auto_config --config data.yml
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

### Maximizing the FP16 performance

Some extra steps may be required to ensure good FP16 performance:

* Mixed precision training requires at least Volta GPUs and CUDA 9.1. As TensorFlow versions 1.5 to 1.12 are shipped with CUDA 9.0, you should instead:
  * install TensorFlow 1.13 or higher
  * use NVIDIA's [TensorFlow Docker image](https://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/running.html)
  * compile TensorFlow manually
* Tensor Cores require the input dimensions to be a multiple of 8. You may need to tune your vocabulary size using `--size_multiple 8` on `onmt-build-vocab` which will ensure that `(vocab_size + 1) % 8 == 0` (+ 1 is the `<unk>` token that is automatically added during the training).

For more information about the implementation and get additional expert recommendation on how to maximize performance, see the [OpenSeq2Seq's documentation](https://nvidia.github.io/OpenSeq2Seq/html/mixed-precision.html).

### Converting between FP32 and FP16

If you want to convert an existing checkpoint to FP16 from FP32 (or vice-versa), see the script `onmt-convert-checkpoint`. Typically, it is useful when you want to train using FP16 but still release a model in FP32, e.g.:

```bash
onmt-convert-checkpoint --model_dir ende-fp16/ --output_dir ende-fp32/ --target_dtype float32
```

The checkpoint generated in `ende-fp32/` can then be used in `infer` or `export` run types.

## Retraining

### Continuing from a stopped training

This is the most common case of retrainings: the training was interrupted but should run longer. In that case, simply **launch the same command that you used for the initial training**, e.g.:

```bash
# Start the training.
onmt-main train_and_eval --model_type NMTSmall --auto_config --config data.yml

# ... the training is interrupted or stopped ...

# Continue from the latest checkpoint.
onmt-main train_and_eval --model_type NMTSmall --auto_config --config data.yml
```

**Note:** If the train was stopped because `train_steps` was reached, you should first increase this value before continuing.

### Fine-tune an existing model

Retraining can also be useful to fine-tune an existing model. For example in machine translation, it is faster to adapt a generic model to a specific domain compared to starting a training from scratch.

OpenNMT-tf offers some features to make this process easier:

* The script `onmt-update-vocab` can be used to change the word vocabularies contained in a checkpoint while keeping learned weights of shared words (e.g. to add a domain terminology)
* The command line argument `--checkpoint_path` can be used to load the weights of an existing checkpoint while starting from a fresh training state (i.e. with new learning rate schedule and optimizer variables)
