# Vocabulary

For text inputs, vocabulary files should be provided in the data configuration. A vocabulary file is a simple text file with **one token per line**. It should start with these 3 special tokens:

```text
<blank>
<s>
</s>
```

## Building vocabularies

The `onmt-build-vocab` script can be used to generate vocabulary files in multiple ways:

### Generate a vocabulary from tokenized training files

If your training data is already tokenized, you can build a vocabulary with the most frequent tokens. For example, the command below extracts the 50,000 most frequent tokens from the files `train.txt.tok` and `other.txt.tok` and saves them to `vocab.txt`:

```bash
onmt-build-vocab --save_vocab vocab.txt --size 50000 train.txt.tok other.txt.tok
```

Instead of defining a fixed size, you can also prune tokens that appear below a minimum frequency. See the `--min_frequency` option.

### Generate a vocabulary from raw training files with on-the-fly tokenization

By default, `onmt-build-vocab` splits each line on spaces. It is possible to define a custom tokenization with the `--tokenizer_config` option. See [Tokenization](tokenization.md) for more information.

### Convert a SentencePiece vocabulary to OpenNMT-tf

If you trained a [SentencePiece](https://github.com/google/sentencepiece) model, a vocabulary file was generated in the process. You can convert this vocabulary to work with OpenNMT-tf:

```bash
onmt-build-vocab --from_vocab sp.vocab --from_format sentencepiece --save_vocab vocab.txt
```

### Train a SentencePiece model and vocabulary with OpenNMT-tf

The `onmt-build-vocab` script can also train a new [SentencePiece](https://github.com/google/sentencepiece) vocabulary and model from raw data. For example the command:

```bash
onmt-build-vocab --sentencepiece --size 32000 --save_vocab sp train.txt.raw
```

will produce the SentencePiece model `sp.model` and the vocabulary `sp.vocab` of size 32,000. The vocabulary file is saved in the OpenNMT-tf format and can be directly used for training.

Additional SentencePiece [training options](https://github.com/google/sentencepiece/blob/master/src/spm_train_main.cc) can be passed to the `--sentencepiece` argument in the format `option=value`, e.g.:

```bash
onmt-build-vocab --sentencepiece character_coverage=0.98 num_threads=4 [...]
```

## Configuring vocabularies

In most cases, you should configure vocabularies with `source_vocabulary` and `target_vocabulary` in `data` block of the YAML configuration, for example:

```yaml
data:
  source_vocabulary: src_vocab.txt
  target_vocabulary: tgt_vocab.txt
```

However, some models may require a different configuration:

* Language models require a single vocabulary:

```yaml
data:
  vocabulary: vocab.txt
```

* Parallel inputs require indexed vocabularies:

```yaml
data:
  source_1_vocabulary: src_1_vocab.txt  # Vocabulary of the 1st source input.
  source_2_vocabulary: src_2_vocab.txt  # Vocabulary of the 2nd source input.
```

* Nested parallel inputs require an additional level of indexing:

```yaml
data:
  source_1_1_vocabulary: src_1_1_vocab.txt
  source_1_2_vocabulary: src_1_2_vocab.txt
  source_2_vocabulary: src_2_vocab.txt
```

**Note:** If you train a model with shared embeddings, you should still configure all vocabulary parameters but in this case they should simply point to the same vocabulary file.
