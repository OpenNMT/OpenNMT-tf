# Transitioning to 2.0

This document details some changes introduced in OpenNMT-tf 2.0 and actions required by the user.

## New requirements

TensorFlow 2.0 and Python 3.5 (or above) are now required by OpenNMT-tf. While the code is still expected to run with Python 2.7, this can change at any time and without notice.

## Changed checkpoint layout

TensorFlow 2.0 introduced a new way to save checkpoints: variables are no longer matched by their name but by where they are stored relative to the model object.

Consequently, all OpenNMT-tf V1 checkpoints are no longer compatible without conversion. To smooth this transition, V1 checkpoints of Transformer-based models are automatically upgraded on load.

## Improved main script command line

The command line parser has been improved to better manage task specifc options that are now located after the run type:

```text
onmt-main <general options> train <train options>
```

Also, some options has been changed:

* the run type `train_and_eval` has been replaced by `train --with_eval`
* the main script now includes the `average_checkpoints` and `update_vocab` tasks
* distributed training options are currently missing in 2.0
* `--session_config` has been removed as no longer applicable in TensorFlow 2.0

See `onmt-main -h` for more details.

## Changed vocabulary configuration

In OpenNMT-tf V1, models were responsible to declare the name of the vocabulary to look for, e.g.:

```python
source_inputter=onmt.inputters.WordEmbedder(
    vocabulary_file_key="source_words_vocabulary",
    embedding_size=512)
```

meant that the user should configure the vocabulary like this:

```yaml
data:
  source_words_vocabulary: src_vocab.txt
```

This is no longer the case in V2 and vocabulary configuration now follows a more general pattern:

* Single vocabulary (e.g. language model):

```yaml
data:
  vocabulary: vocab.txt
```

* Source and target vocabularies (e.g. sequence to sequence, tagging, etc.):

```yaml
data:
  source_vocabulary: src_vocab.txt
  target_vocabulary: tgt_vocab.txt
```

* Multi-source and target vocabularies:

```yaml
data:
  source_1_vocabulary: src_1_vocab.txt
  source_2_vocabulary: src_2_vocab.txt
  target_vocabulary: tgt_vocab.txt
```

The same name resolution applies to "embedding" and "tokenization" configurations.

## Renamed parameters

Some parameters in the YAML have been renamed or removed:

| V1 | V2 | Comment |
| --- | --- | --- |
| `*/bucket_width` | `*/length_bucket_width` | |
| `*/num_threads` | | Automatic value |
| `*/prefetch_buffer_size` | | Automatic value |
| `eval/eval_delay` | `eval/steps` | Use steps instead of seconds to set the evaluation frequency |
| `eval/exporters` | | Not implemented |
| `params/decay_step_duration` | | No longer useful |
| `params/freeze_variables` | `params/freeze_layers` | Layer names instead of variable regexps |
| `params/gradients_accum` | | Use `train/effective_batch_size` instead |
| `params/loss_scale` | | Dynamic loss scaling by default |
| `params/maximum_iterations` | `maximum_decoding_length` | |
| `params/maximum_learning_rate` | | Not implemented |
| `params/param_init` | | Not implemented |
| `params/weight_decay` | | Not implemented |
| `train/save_checkpoints_secs` | | Not implemented |
| `train/train_steps` | `train/max_step` | |

Parameters taking reference to Python classes should also be revised when upgrading to V2:

* `params/optimizer`
* `params/optimizer_params`
* `params/decay_type`
* `params/decay_params`

## New mixed precision workflow

OpenNMT-tf 2.0 is using the newly introduced [graph rewriter](https://github.com/tensorflow/tensorflow/pull/26342) to automatically convert parts of the graph from *float32* to *float16*.

Variables are casted on the fly and checkpoints no longer need to be converted for inference or to continue training in *float32*. This means mixed precision is no longer an property of the model but should be enabled on the command line instead, e.g.:

```bash
onmt-main --model_type Transformer --config data.yml --auto_config --mixed_precision train
```
