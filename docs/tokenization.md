# Tokenization

By default, OpenNMT-tf **expects and generates tokenized text**. The user is thus responsible to tokenize the input and detokenize the output. However, OpenNMT-tf provides tokenization tools based on the C++ OpenNMT [Tokenizer](https://github.com/OpenNMT/Tokenizer) that can be used in 2 ways:

* *offline*: use the provided scripts to manually tokenize the text files before the execution and detokenize the output for evaluation
* *online*: configure the execution to apply tokenization and detokenization on-the-fly

*Note: the `pyonmttok` package is only supported on Linux as of now.*

## Configuration files

YAML files are used to set the tokenizer options to ensure consistency during data preparation and training. For example:

```yaml
mode: aggressive
joiner_annotate: true
segment_numbers: true
segment_alphabet_change: true
```

*For a complete list of available options, see the [Tokenizer documentation](https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md).*

## Offline usage

You can invoke the `onmt-tokenize-text` script directly and select the `OpenNMTTokenizer` tokenizer:

```bash
$ echo "Hello world!" | onmt-tokenize-text --tokenizer OpenNMTTokenizer
Hello world !
$ echo "Hello world!" | onmt-tokenize-text --tokenizer OpenNMTTokenizer --tokenizer_config config/tokenization/aggressive.yml
Hello world ￭!
```

## Online usage

A key feature is the possibility to tokenize the data on-the-fly during the training. This avoids the need of storing tokenized files and also increases the consistency of your preprocessing pipeline.

Here is an example workflow:

1\. Build the vocabularies with the custom tokenizer, e.g.:

```bash
onmt-build-vocab --tokenizer OpenNMTTokenizer --tokenizer_config config/tokenization/aggressive.yml --size 50000 --save_vocab data/enfr/en-vocab.txt data/enfr/en-train.txt
onmt-build-vocab --tokenizer OpenNMTTokenizer --tokenizer_config config/tokenization/aggressive.yml --size 50000 --save_vocab data/enfr/fr-vocab.txt data/enfr/fr-train.txt
```

*The text files are only given as examples and are not part of the repository.*

2\. Update your model's `TextInputter`s to use the custom tokenizer, e.g.:

```python
return onmt.models.SequenceToSequence(
    source_inputter=onmt.inputters.WordEmbedder(
        vocabulary_file_key="source_words_vocabulary",
        embedding_size=512,
        tokenizer=onmt.tokenizers.OpenNMTTokenizer(
            configuration_file_or_key="source_tokenizer_config")),
    target_inputter=onmt.inputters.WordEmbedder(
        vocabulary_file_key="target_words_vocabulary",
        embedding_size=512,
        tokenizer=onmt.tokenizers.OpenNMTTokenizer(
            configuration_file_or_key="target_tokenizer_config")),
    ...)
```

3\. Reference the tokenizer configurations in the data configuration, e.g.:

```yaml
data:
  source_tokenizer_config: config/tokenization/aggressive.yml
  target_tokenizer_config: config/tokenization/aggressive.yml
```

## Notes

* As of now, tokenizers are not part of the exported graph. However, all tokenization resources (configuration, subword models, etc.) are registered as additional assets in the `SavedModel` bundle in the `assets.extra` directory.
* Predictions saved during inference or evaluation are detokenized. Consider using the "BLEU-detok" external evaluator that calls `multi-bleu-detok.perl` instead of `multi-bleu.perl`.
