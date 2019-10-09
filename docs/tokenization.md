# Tokenization

By default, OpenNMT-tf **expects and generates tokenized text**. The users are thus responsible to tokenize the input and detokenize the output with the tool of their choice.

However, OpenNMT-tf provides tokenization tools based on the C++ OpenNMT [Tokenizer](https://github.com/OpenNMT/Tokenizer) that can be used in 2 ways:

* *offline*: use the provided scripts to manually tokenize the text files before the execution and detokenize the output for evaluation
* *online*: configure the execution to apply tokenization and detokenization on-the-fly

*Note: the `pyonmttok` package is only supported on Linux as of now.*

## Configuration files

YAML files are used to set the tokenizer options to ensure consistency during data preparation and training. For example, this configuration defines a simple word-based tokenization using the OpenNMT tokenizer:

```yaml
type: OpenNMTTokenizer
params:
  mode: aggressive
  joiner_annotate: true
  segment_numbers: true
  segment_alphabet_change: true
```

*For a complete list of available options, see the <a href="https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md">Tokenizer documentation</a>).*

OpenNMT-tf also defines additional tokenizers:

* `CharacterTokenizer`
* `SpaceTokenizer`

## Offline usage

You can invoke the `onmt-tokenize-text` script directly and pass the tokenizer configuration:

```bash
$ echo "Hello world!" | onmt-tokenize-text --tokenizer_config config/tokenization/aggressive.yml
Hello world ￭!
```

## Online usage

A key feature is the possibility to tokenize the data on-the-fly during the training. This avoids the need of storing tokenized files and also increases the consistency of your preprocessing pipeline.

Here is an example workflow:

1\. Build the vocabularies with the custom tokenizer, e.g.:

```bash
onmt-build-vocab --tokenizer_config config/tokenization/aggressive.yml --size 50000 --save_vocab data/enfr/en-vocab.txt data/enfr/en-train.txt
onmt-build-vocab --tokenizer_config config/tokenization/aggressive.yml --size 50000 --save_vocab data/enfr/fr-vocab.txt data/enfr/fr-train.txt
```

*The text files are only given as examples and are not part of the repository.*

2\. Reference the tokenizer configurations in the data configuration, e.g.:

```yaml
data:
  source_tokenization: config/tokenization/aggressive.yml
  target_tokenization: config/tokenization/aggressive.yml
```

## Exported graph

Only TensorFlow ops can be exported to graphs and used for serving. When a tokenizer is not implemented in terms of TensorFlow ops such as the OpenNMT tokenizer, it will not be part of the exported graph. The model will then expects tokenized inputs during serving.

**In-graph tokenizers:**

* `CharacterTokenizer`
* `SpaceTokenizer`

Model inputs: `text` (1D string tensor)

**Out-of-graph tokenizers:**

* `OpenNMTTokenizer` (\*)

Model inputs: `tokens` (2D string tensor), `length` (1D int32 tensor)

(\*) During model export, tokenization resources used by the OpenNMT tokenizer (configuration, subword models, etc.) are registered as additional assets in the `SavedModel`'s `assets.extra` directory.
