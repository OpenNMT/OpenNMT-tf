# Training

```bash
onmt-main train -h
```

**Example:**

```bash
onmt-main --model_type Transformer --config data.yml --auto_config train --with_eval
```

## Monitoring

OpenNMT-tf uses [TensorBoard](https://www.tensorflow.org/tensorboard) to log information during the training. Simply start `tensorboard` by setting the active log directory, e.g.:

```bash
tensorboard --logdir="."
```

then open the URL displayed in the shell to monitor and visualize several data, including:

* training and evaluation loss
* training speed
* learning rate
* gradients norm
* word embeddings
* decoder sampling probability

## Automatic evaluation

Evaluation can be run automatically when using the `--with_eval` flag:

```bash
onmt-main [...] train --with_eval
```

This minimally requires you to set some evaluation files in your data configuration, e.g.:

```yaml
data:
  eval_features_file: ...
  eval_labels_file: ...
```

By default, it will run every 5000 training steps and report evaluation results in the console output and on TensorBoard.

### Export model on best metric

Automatic evaluation can also export an inference model when a metric reaches its best value so far. For example, the following configuration will make the training exports a model each time the evaluation scores the best BLEU score so far:

```yaml
eval:
  scorers: bleu
  export_on_best: bleu
```

These models are saved in the model directory under `export/<step>`. See also [Serving](serving.md) for more information about exported models.

### Early stopping

Early stopping is useful to stop the training automatically when the model performance is not improving anymore.

For example, the following configuration stops the training when the BLEU score does not improve by more than 0.2 points in the last 4 evaluations:

```yaml
eval:
  scorers: bleu
  early_stopping:
    metric: bleu
    min_improvement: 0.2
    steps: 4
```

## Multi-GPU training

OpenNMT-tf training can run on multiple GPUs. For example if your machine has 4 GPUs, simply add `--num_gpus 4` to the training command line:

```bash
onmt-main [...] train --num_gpus 4
```

Multi-GPU training uses data parallelism. Each GPU computes the gradients for a different batch and then gradients are reduced across all devices. The reduced gradients are used to update the model parameters. It is equivalent to train with batches `N` times larger, where `N` is the number of used GPUs.

Note that evaluation and inference will run on a single device.

## Distributed training with Horovod

Multi-GPU and multi-node trainings are also supported with [Horovod](https://github.com/horovod/horovod), a popular and generic distributed training framework. Horovod can offer better performance and customization than the TensorFlow distributed framework.

For example, the command below will start a local training on 4 GPUs (note the `--horovod` training flag):

```bash
horovodrun -np 4 -H localhost:4 onmt-main [...] train --horovod
```

Similar to multi-GPU training, this is equivalent to train with batches `N` times larger, where `N` is the total number of Horovod processes.

See the [Horovod documentation](https://horovod.readthedocs.io/en/stable/index.html) for more information about installation and usage.

## Mixed precision training

Mixed precision can be enabled with the `--mixed_precision` flag:

```bash
onmt-main --model_type Transformer --auto_config --config data.yml --mixed_precision train
```

This mode enables the Keras "mixed\_float16" policy where layers use float16 computations and float32 variables. During training, dynamic loss scaling is used to prevent underflow in intermediate gradients. See the TensorFlow [mixed precision guide](https://www.tensorflow.org/guide/mixed_precision) for more information.

### Maximizing the FP16 performance

Some extra steps may be required to ensure good FP16 performance:

* Mixed precision training requires a Volta GPU or above
* Tensor Cores require the input dimensions to be a multiple of 8. You may need to tune your vocabulary size using `--size_multiple 8` on `onmt-build-vocab` which will ensure that `(vocab_size + 1) % 8 == 0` (+ 1 is the `<unk>` token that is automatically added during the training).

## Retraining

### Continuing from a stopped training

This is the most common case of retrainings: the training was interrupted but should run longer. In that case, simply **launch the same command that you used for the initial training**, e.g.:

```bash
# Start the training.
onmt-main --model_type Transformer --config data.yml --auto_config train

# ... the training is interrupted or stopped ...

# Continue from the latest checkpoint.
onmt-main --model_type Transformer --config data.yml --auto_config train
```

**Note:** If the train was stopped because `max_step` was reached, you should first increase this value before continuing.

If you want to continue the training in another model directory, do the following:

1. Update `model_dir` in the configuration to the new model directory
2. Continue training from the previous model directory using the options `--checkpoint_path` and `--continue_from_checkpoint`:

```bash
onmt-main [...] --checkpoint_path /path/to/previous/model/dir train --continue_from_checkpoint
```

### Fine-tune an existing model

Retraining can also be useful to fine-tune an existing model. For example in machine translation, it is faster to adapt a generic model to a specific domain compared to starting a training from scratch.

For a basic domain adaptation, we recommend tokenizing your data with a subword segmentation model (such as [SentencePiece](https://github.com/google/sentencepiece)) and simply continue the training on new data.

However, OpenNMT-tf also offers some features that could be useful in more advanced workflows:

* The run type `update_vocab` can be used to change the word vocabularies contained in a checkpoint while keeping learned weights of shared words (e.g. to add a domain terminology)
* The command line argument `--checkpoint_path` can be used to load the weights of an existing checkpoint while starting from a fresh training state (i.e. with new learning rate schedule and optimizer variables)
* The parameter `freeze_layers` in the YAML configuration can be used to prevent the adapation of specific layers in the model
