# Frequently Asked Questions

## How to count the number of epochs?

OpenNMT-tf uses a step-based training that makes it difficult to track the number of epochs over the dataset. However, it is possible to run the training epoch by epoch with the following configuration:

```yaml
train:
  max_step: null
  single_pass: true
```

and then:

```text
onmt-main [...] train  # 1st epoch
onmt-main [...] train  # 2nd epoch
...
```

which you can wrap in a shell loop for example.

## How to continue training with different optimization settings?

By default, OpenNMT-tf continues the training where it left off, including the state of the optimizer. To change the optimization settings, the recommended approach is to start a fresh training and only load the model weights from the previous checkpoint:

1. Set a new model directory, either with the command line option `--model_dir` or in the configuration field `model_dir`
2. Set the command line option `--checkpoint_path` to the checkpoint where model weights should be loaded

## How can I restrict the TensorFlow runtime to specific GPU?

Use the [`CUDA_VISIBLE_DEVICES`](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) environment variable, e.g.:

```
CUDA_VISIBLE_DEVICES=0,1 onmt-main [...]
```
