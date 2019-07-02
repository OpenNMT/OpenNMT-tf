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
onmt-main [...] train --num_gpus 4
```

Note that evaluation and inference will run on a single device.

## Mixed precision training

Thanks to [work from NVIDIA](https://github.com/NVIDIA/OpenSeq2Seq), OpenNMT-tf supports training models using FP16 computation. Mixed precision training is automatically enabled when the data type of the [inputters](package/opennmt.inputters.inputter.html) is defined to be `tf.float16`. See for example the predefined model `TransformerFP16`, which is up to [1.8x faster](https://github.com/OpenNMT/OpenNMT-tf/pull/211#issuecomment-455605090) than the FP32 version on compatible hardware:

```bash
onmt-main --model_type TransformerFP16 --auto_config --config data.yml train
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
onmt-main --model_type NMTSmall --auto_config --config data.yml train

# ... the training is interrupted or stopped ...

# Continue from the latest checkpoint.
onmt-main --model_type NMTSmall --auto_config --config data.yml train
```

**Note:** If the train was stopped because `train_steps` was reached, you should first increase this value before continuing.

### Fine-tune an existing model

Retraining can also be useful to fine-tune an existing model. For example in machine translation, it is faster to adapt a generic model to a specific domain compared to starting a training from scratch.

OpenNMT-tf offers some features to make this process easier:

* The script `onmt-update-vocab` can be used to change the word vocabularies contained in a checkpoint while keeping learned weights of shared words (e.g. to add a domain terminology)
* The command line argument `--checkpoint_path` can be used to load the weights of an existing checkpoint while starting from a fresh training state (i.e. with new learning rate schedule and optimizer variables)
