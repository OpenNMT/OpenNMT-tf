# Tokenization

OpenNMT-tf can use the OpenNMT [Tokenizer](https://github.com/OpenNMT/Tokenizer) as a plugin to provide advanced tokenization behaviors.

## Installation

*Note: we plan to release a package on PyPi to facilitate the installation of this plugin.*

1\. Follow the Tokenizer [installation instructions](https://github.com/OpenNMT/Tokenizer) with [Python bindings](https://github.com/OpenNMT/Tokenizer/tree/master/bindings/python).

2\. Configure your environment for Python to find the newly generated package, e.g.:

```bash
export PYTHONPATH="$PYTHONPATH:$HOME/Tokenizer/build/bindings/python/"
```

3\. Test the plugin:

```bash
$ echo "Hello world!" | onmt-tokenize-text --tokenizer OpenNMTTokenizer
Hello world !
```

## Usage

YAML files are used to set the tokenizer options to ensure consistency during data preparation and training. See the sample file `config/tokenization/sample.yml`.

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

* As of now, tokenizers are not part of the exported graph.
* Predictions saved during inference or evaluation are detokenized. Consider using the "BLEU-detok" external evaluator that calls `multi-bleu-detok.perl` instead of `multi-bleu.perl`.
