# 2.0 Transition Guide

This document details some changes introduced in OpenNMT-tf 2.0 and actions required by the user.

## New requirements

### Python 3

Python 3.5 (or above) is now required by OpenNMT-tf. See [python3statement.org](https://python3statement.org/) for more context about this decision.

### TensorFlow 2.0

OpenNMT-tf has been completely redesigned for TensorFlow 2.0 which is now the minimal required TensorFlow version.

The correct TensorFlow version is declared as a dependency of OpenNMT-tf and will be automatically installed as part of the `pip` installation:

```bash
pip install --upgrade pip
pip install OpenNMT-tf
```

## Changed checkpoint layout

TensorFlow 2.0 introduced a new way to save checkpoints: variables are no longer matched by their name but by where they are stored relative to the root model object. Consequently, OpenNMT-tf V1 checkpoints are no longer compatible without conversion.

To smooth this transition, V1 checkpoints of the following models are automatically upgraded on load:

* NMTBigV1
* NMTMediumV1
* NMTSmallV1
* Transformer
* TransformerBig

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
# V1 model inputter.
source_inputter=onmt.inputters.WordEmbedder(
    vocabulary_file_key="source_words_vocabulary",
    embedding_size=512)
```

meant that the user should configure the vocabulary like this:

```yaml
# V1 vocabulary configuration.
data:
  source_words_vocabulary: src_vocab.txt
```

This is no longer the case in V2 where vocabulary configuration now follows a general pattern, the same that is currently used for "embedding" and "tokenization" configurations.

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
```

* Nested inputs:

```yaml
data:
  source_1_1_vocabulary: src_1_1_vocab.txt
  source_1_2_vocabulary: src_1_2_vocab.txt
  source_2_vocabulary: src_2_vocab.txt
```

## Changed predefined models

Predefined models do not require a model definition file and can be directly set to the `--model_type` command line argument. Some of them have been changed for clarity:

| V1 | V2 | Comment |
| --- | --- | --- |
| `NMTBig` | `NMTBigV1` | |
| `NMTMedium` | `NMTMediumV1` | |
| `NMTSmall` | `NMTSmallV1` | |
| `SeqTagger` | `LstmCnnCrfTagger` | |
| `TransformerAAN` | | Not considered useful compared to the standard Transformer |
| `TransformerBigFP16` | | Use `TransformerBig` with `--mixed_precision` flag on the command line |
| `TransformerFP16` | | Use `Transformer` with `--mixed_precision` flag on the command line |

## Changed parameters

Some parameters in the YAML configuration have been renamed or removed:

| V1 | V2 | Comment |
| --- | --- | --- |
| `*/bucket_width` | `*/length_bucket_width` | |
| `*/num_threads` | | Automatic value |
| `*/prefetch_buffer_size` | | Automatic value |
| `eval/eval_delay` | `eval/steps` | Use steps instead of seconds to set the evaluation frequency |
| `eval/exporters` | | Not implemented |
| `params/clip_gradients` | | Set [`clipnorm` or `clipvalue`](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#__init__) in `params/optimizer_params/` |
| `params/decay_step_duration` | | No longer useful |
| `params/freeze_variables` | `params/freeze_layers` | Use layer names instead of variable regexps |
| `params/gradients_accum` | | Use `train/effective_batch_size` instead |
| `params/horovod` | | Not implemented |
| `params/loss_scale` | | Dynamic loss scaling by default |
| `params/maximum_iterations` | `params/maximum_decoding_length` | |
| `params/maximum_learning_rate` | | Not implemented |
| `params/param_init` | | Not implemented |
| `params/weight_decay` | `params/optimizer_params/weight_decay` | |
| `train/save_checkpoints_secs` | | Not implemented |
| `train/train_steps` | `train/max_step` | |

Parameters taking reference to Python classes should also be revised when upgrading to V2, as the class likely changed in the process. This concerns:

* `params/optimizer` and `params/optimizer_params`
* `params/decay_type` and `params/decay_params`

## New mixed precision workflow

OpenNMT-tf 2.0 is using the newly introduced [graph rewriter](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/train/experimental/enable_mixed_precision_graph_rewrite) to automatically convert parts of the graph from `float32` to `float16`.

Variables are casted on the fly and checkpoints no longer need to be converted for inference or to continue training in `float32`. This means mixed precision is no longer a property of the model but should be enabled on the command line instead, e.g.:

```bash
onmt-main --model_type Transformer --config data.yml --auto_config --mixed_precision train
```
