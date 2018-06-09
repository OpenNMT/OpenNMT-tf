# Inference

## Checkpoints averaging

The script `onmt-average-checkpoints` can be used to average the parameters of several checkpoints, usually increasing the model performance. For example:

```bash
onmt-average-checkpoints \
    --model_dir run/baseline-enfr \
    --output_dir run/baseline-enfr/avg \
    --max_count 5
```

will average the parameters of the 5 latest checkpoints in the `run/baseline-enfr` model directory and save a new checkpoint in the directory `run/baseline-enfr/avg`.

Then, execute the inference by setting the `--checkpoint_path` option, e.g.:

```bash
onmt-main infer \
    --config config/my_config.yml \
    --features_file newstest2014.en.tok \
    --predictions_file newstest2014.en.tok.out \
    --checkpoint_path run/baseline-enfr/avg/model.ckpt-200000
```

To control the saving of checkpoints during the training, configure the following options in your configuration file:

```yaml
train:
  # (optional) Save a checkpoint every this many steps.
  save_checkpoints_steps: 5000
  # (optional) How many checkpoints to keep on disk.
  keep_checkpoint_max: 10
```

## N-best list

A n-best list can be generated for models using beam search. You can configure it in your configuration file:

```yaml
infer:
  n_best: 5
```

With this option, each input line will simply generate N consecutive lines in the output, ordered from best to worst.

*Note that N can not be greater than the configured `beam_width`.*

## Scoring

The main OpenNMT-tf script can also be used to score existing translations via the `score` run type. It requires 2 command line options to be set:

* `--features_file`, the input labels;
* `--predictions_file`, the translations to score.

e.g.:

```bash
onmt-main score \
    --config config/my_config.yml \
    --features_file newstest2014.en.tok \
    --predictions_file newstest2014.en.tok.out
```

The command will write on the standard output the score generated for each line in the following format:

```text
<score> ||| <translation>
```

where `<score>` is the negative log likelihood of the provided translation.

**Tip:** combining the n-best list generation and the scoring can be used for reranking translations.
