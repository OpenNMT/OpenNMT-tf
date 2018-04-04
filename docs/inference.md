# Inference

## Checkpoints averaging

The script `onmt-average-checkpoints` can be used to average the parameters of several checkpoints, usually increasing the model performance. For example:

```bash
onmt-average-checkpoints --model_dir run/baseline-enfr --output_dir run/baseline-enfr/avg --max_count 5
```

will average the parameters of the 5 latest checkpoints in the `run/baseline-enfr` model directory and save a new checkpoint in the directory `run/baseline-enfr/avg`.

Then, execute the inference by setting the `--checkpoint_path` option, e.g.:

```bash
onmt-main infer --config config/my_config.yml --features_file newstest2014.en.tok --predictions_file newstest2014.en.tok.out --checkpoint_path run/baseline-enfr/avg/model.ckpt-200000
```

To control the saving of checkpoints during the training, configure the following options in your configuration file:

```yml
train:
  # (optional) Save a checkpoint every this many steps.
  save_checkpoints_steps: 5000
  # (optional) How many checkpoints to keep on disk.
  keep_checkpoint_max: 10
```
