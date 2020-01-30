# Vocabulary

For text inputs, vocabulary files should be provided in the data configuration. A vocabulary file is a simple text file with **one token per line**. It shoud starts with these 3 special tokens:

```text
<blank>
<s>
</s>
```

## Building vocabularies

The `onmt-build-vocab` script can be used to generate vocabulary files in multiple ways:

### from tokenized files

The script can be run directly on tokenized files. For example the command:

```bash
onmt-build-vocab --size 50000 --save_vocab vocab.txt train.txt.tok
```

extracts the 50,000 most frequent tokens from `train.txt.tok` and saves them to `vocab.txt`.

### from raw files

By default, `onmt-build-vocab` splits each line on spaces. It is possible to define a custom tokenization with the `--tokenizer_config` option. See [Tokenization](tokenization.md) for more information.

### from SentencePiece training

`onmt-build-vocab` can also prepare a SentencePiece vocabulary and model from raw data. For example the command:

```bash
onmt-build-vocab --sentencepiece --size 32000 --save_vocab sp train.txt.raw
```

will produce the SentencePiece model `sp.model` and the vocabulary `sp.vocab` of size 32,000. Additional SentencePiece [training options](https://github.com/google/sentencepiece/blob/master/src/spm_train_main.cc) can be passed to the `--sentencepiece` argument in the format `option=value`, e.g. `--sentencepiece character_coverage=0.98 num_threads=4`.

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
```
