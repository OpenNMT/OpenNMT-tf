# Inference

## Checkpoints averaging

The `average_checkpoints` run type can be used to average the parameters of several checkpoints, usually increasing the model performance. For example:

```bash
onmt-main \
    --config config/my_config.yml --auto_config \
    average_checkpoints \
    --output_dir run/baseline-enfr/avg \
    --max_count 5
```

will average the parameters of the 5 latest checkpoints from the model directory configured in `config/my_config.yml` and save a new checkpoint in the directory `run/baseline-enfr/avg`.

Then, execute the inference by setting the `--checkpoint_path` option, e.g.:

```bash
onmt-main \
    --config config/my_config.yml --auto_config \
    --checkpoint_path run/baseline-enfr/avg/ckpt-200000 \
    infer --features_file newstest2014.en.tok --predictions_file newstest2014.en.tok.out
```

To control the saving of checkpoints during the training, configure the following options in your configuration file:

```yaml
train:
  # (optional) Save a checkpoint every this many steps.
  save_checkpoints_steps: 5000
  # (optional) How many checkpoints to keep on disk.
  keep_checkpoint_max: 10
```

## Random sampling

Sampling predictions from the output distribution can be an effective decoding strategy for back-translation, as described by [Edunov et al. 2018](https://arxiv.org/abs/1808.09381). To enable this feature, you should configure the parameter `sampling_topk`. Possible values are:

* `k`, sample from the `k` most likely tokens
* `0`, sample from the full output distribution
* `1`, no sampling (default)

For example:

```yaml
params:
  beam_width: 1
  sampling_topk: 0
  sampling_temperature: 1
```

## Noising

Noising the decoded output is also a possible decoding strategy for back-translation, as described in [Edunov et al. 2018](https://arxiv.org/abs/1808.09381). 3 types of noise are currently implemented:

* [`dropout`](https://opennmt.net/OpenNMT-tf/package/opennmt.data.WordDropout): randomly drop words in the sequence
* [`replacement`](https://opennmt.net/OpenNMT-tf/package/opennmt.data.WordReplacement): randomly replace words by a filler token
* [`permutation`](https://opennmt.net/OpenNMT-tf/package/opennmt.data.WordPermutation): randomly permute words with a maximum distance

which can be combined in sequence, e.g.:

```yaml
params:
  decoding_subword_token: ▁
  decoding_noise:
    - dropout: 0.1
    - replacement: [0.1, ｟unk｠]
    - permutation: 3
```

The parameter `decoding_subword_token` (here set to the SentencePiece spacer) is useful to apply noise at the word level instead of the subword level.

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
onmt-main \
    --config config/my_config.yml --auto_config \
    score
    --features_file newstest2014.en.tok \
    --predictions_file newstest2014.en.tok.out
```

The command will write on the standard output the score generated for each line in the following format:

```text
<score> ||| <translation>
```

where `<score>` is the negative log likelihood of the provided translation.

**Tip:** combining the n-best list generation and the scoring can be used for reranking translations.
