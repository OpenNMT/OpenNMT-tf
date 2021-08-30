# Tokenization

By default, OpenNMT-tf **expects and generates tokenized text**. The users are thus responsible to tokenize the input and detokenize the output with the tool of their choice.

However, OpenNMT-tf integrates several tokenizers that can be used to process the data:

* `SpaceTokenizer` (default): splits on spaces
* `CharacterTokenizer`: segments each character and replaces spaces by special characters
* `OpenNMTTokenizer`: applies the OpenNMT [Tokenizer](https://github.com/OpenNMT/Tokenizer)
* `SentencePieceTokenizer`: applies a SentencePiece tokenization using [tensorflow-text](https://github.com/tensorflow/text)

## Configuration files

YAML files are used to set the tokenizer options to ensure consistency during data preparation and training. They should contain 2 fields:

* `type`: the name of the tokenizer to use
* `params`: the parameters to use for this tokenizer

### Example: BPE tokenization

The configuration below defines a basic BPE tokenization using the OpenNMT [Tokenizer](https://github.com/OpenNMT/Tokenizer):

```yaml
type: OpenNMTTokenizer
params:
  mode: aggressive
  bpe_model_path: /path/to/bpe.model
  joiner_annotate: true
  segment_numbers: true
  segment_alphabet_change: true
  preserve_segmented_tokens: true
```

*For a complete list of available options, see the <a href="https://github.com/OpenNMT/Tokenizer/blob/master/docs/options.md">Tokenizer documentation</a>.*

### Example: SentencePiece tokenization

The `SentencePieceTokenizer` applies a [SentencePiece](https://github.com/google/sentencepiece) tokenization using [tensorflow-text](https://github.com/tensorflow/text) (make sure to install this package to use this tokenizer).

```yaml
type: SentencePieceTokenizer
params:
  model: /path/to/sentencepiece.model
```

This tokenizer is implemented as a TensorFlow op so it is included in the exported graph (see [Exported graph](#exported-graph)).

## Applying the tokenization

### Offline

The tokenization can be applied before starting the training using the script `onmt-tokenize-text`. The tokenizer configuration should be passed as argument:

```bash
$ echo "Hello world!" | onmt-tokenize-text --tokenizer_config config/tokenization/aggressive.yml
Hello world ￭!
```

The script `onmt-detokenize-text` can later be used for detokenization:

```bash
$ echo "Hello world ￭!" | onmt-detokenize-text --tokenizer_config config/tokenization/aggressive.yml
Hello world!
```

### Online

A key feature is the possibility to tokenize the data on-the-fly during training and inference. This avoids the need of storing tokenized files and also increases the consistency of your preprocessing pipeline.

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
* `SentencePieceTokenizer`
* `SpaceTokenizer`

Model inputs: `text` (1D string tensor)

**Out-of-graph tokenizers:**

* `OpenNMTTokenizer` (\*)

Model inputs: `tokens` (2D string tensor), `length` (1D int32 tensor)

(\*) During model export, tokenization resources used by the OpenNMT tokenizer (configuration, subword models, etc.) are registered as additional assets in the `SavedModel`'s `assets.extra` directory.
