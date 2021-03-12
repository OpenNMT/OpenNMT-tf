[![CI](https://github.com/OpenNMT/OpenNMT-tf/workflows/CI/badge.svg)](https://github.com/OpenNMT/OpenNMT-tf/actions?query=workflow%3ACI) [![codecov](https://codecov.io/gh/OpenNMT/OpenNMT-tf/branch/master/graph/badge.svg)](https://codecov.io/gh/OpenNMT/OpenNMT-tf) [![PyPI version](https://badge.fury.io/py/OpenNMT-tf.svg)](https://badge.fury.io/py/OpenNMT-tf) [![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://opennmt.net/OpenNMT-tf/) [![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-tf.svg)](https://gitter.im/OpenNMT/OpenNMT-tf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

# OpenNMT-tf

OpenNMT-tf is a general purpose sequence learning toolkit using TensorFlow 2. While neural machine translation is the main target task, it has been designed to more generally support:

* sequence to sequence mapping
* sequence tagging
* sequence classification
* language modeling

The project is production-oriented and comes with [backward compatibility guarantees](https://github.com/OpenNMT/OpenNMT-tf/blob/master/CHANGELOG.md).

## Key features

### Modular model architecture

Models are described with code to allow training custom architectures and overriding default behavior. For example, the following instance defines a sequence to sequence model with 2 concatenated input features, a self-attentional encoder, and an attentional RNN decoder sharing its input and output embeddings:

```python
opennmt.models.SequenceToSequence(
    source_inputter=opennmt.inputters.ParallelInputter(
        [
            opennmt.inputters.WordEmbedder(embedding_size=256),
            opennmt.inputters.WordEmbedder(embedding_size=256),
        ],
        reducer=opennmt.layers.ConcatReducer(axis=-1),
    ),
    target_inputter=opennmt.inputters.WordEmbedder(embedding_size=512),
    encoder=opennmt.encoders.SelfAttentionEncoder(num_layers=6),
    decoder=opennmt.decoders.AttentionalRNNDecoder(
        num_layers=4,
        num_units=512,
        attention_mechanism_class=tfa.seq2seq.LuongAttention,
    ),
    share_embeddings=opennmt.models.EmbeddingsSharingLevel.TARGET,
)
```

The [`opennmt`](https://opennmt.net/OpenNMT-tf/package/opennmt.html) package exposes other building blocks that can be used to design:

* [multiple input features](https://opennmt.net/OpenNMT-tf/package/opennmt.inputters.ParallelInputter.html)
* [mixed embedding representation](https://opennmt.net/OpenNMT-tf/package/opennmt.inputters.MixedInputter.html)
* [multi-source context](https://opennmt.net/OpenNMT-tf/package/opennmt.inputters.ParallelInputter.html)
* [cascaded](https://opennmt.net/OpenNMT-tf/package/opennmt.encoders.SequentialEncoder.html) or [multi-column](https://opennmt.net/OpenNMT-tf/package/opennmt.encoders.ParallelEncoder.html) encoder
* [hybrid sequence to sequence models](https://opennmt.net/OpenNMT-tf/package/opennmt.models.SequenceToSequence.html)

Standard models such as the Transformer are defined in a [model catalog](https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/models/catalog.py) and can be used without additional configuration.

*Find more information about model configuration in the [documentation](https://opennmt.net/OpenNMT-tf/model.html).*

### Full TensorFlow 2 integration

OpenNMT-tf is fully integrated in the TensorFlow 2 ecosystem:

* Reusable layers extending [`tf.keras.layers.Layer`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer)
* Multi-GPU training with [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute) and distributed training with [Horovod](https://github.com/horovod/horovod)
* Mixed precision training with [`tf.keras.mixed_precision`](https://www.tensorflow.org/guide/mixed_precision)
* Visualization with [TensorBoard](https://www.tensorflow.org/tensorboard)
* `tf.function` graph tracing that can be [exported to a SavedModel](https://opennmt.net/OpenNMT-tf/serving.html) and served with [TensorFlow Serving](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving/tensorflow_serving) or [Python](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving/python)

### Compatibility with CTranslate2

[CTranslate2](https://github.com/OpenNMT/CTranslate2) is an optimized inference engine for OpenNMT models featuring fast CPU and GPU execution, model quantization, parallel translations, dynamic memory usage, interactive decoding, and more! OpenNMT-tf can [automatically export](https://opennmt.net/OpenNMT-tf/serving.html#ctranslate2) models to be used in CTranslate2.

### Dynamic data pipeline

OpenNMT-tf does not require to compile the data before the training. Instead, it can directly read text files and preprocess the data when needed by the training. This allows [on-the-fly tokenization](https://opennmt.net/OpenNMT-tf/tokenization.html) and data augmentation by injecting random noise.

### Model fine-tuning

OpenNMT-tf supports model fine-tuning workflows:

* Model weights can be transferred to new word vocabularies, e.g. to inject domain terminology before fine-tuning on in-domain data
* [Contrastive learning](https://ai.google/research/pubs/pub48253/) to reduce word omission errors

### Source-target alignment

Sequence to sequence models can be trained with [guided alignment](https://arxiv.org/abs/1607.01628) and alignment information are returned as part of the translation API.

---

OpenNMT-tf also implements most of the techniques commonly used to train and evaluate sequence models, such as:

* automatic evaluation during the training
* multiple decoding strategy: greedy search, beam search, random sampling
* N-best rescoring
* gradient accumulation
* scheduled sampling
* checkpoint averaging
* ... and more!

*See the [documentation](https://opennmt.net/OpenNMT-tf/) to learn how to use these features.*

## Usage

OpenNMT-tf requires:

* Python 3.5 or above
* TensorFlow 2.3, 2.4

We recommend installing it with `pip`:

```bash
pip install --upgrade pip
pip install OpenNMT-tf
```

*See the [documentation](https://opennmt.net/OpenNMT-tf/installation.html) for more information.*

### Command line

OpenNMT-tf comes with several command line utilities to prepare data, train, and evaluate models.

For all tasks involving a model execution, OpenNMT-tf uses a unique entrypoint: `onmt-main`. A typical OpenNMT-tf run consists of 3 elements:

* the **model** type
* the **parameters** described in a YAML file
* the **run** type such as `train`, `eval`, `infer`, `export`, `score`, `average_checkpoints`, or `update_vocab`

that are passed to the main script:

```
onmt-main --model_type <model> --config <config_file.yml> --auto_config <run_type> <run_options>
```

*For more information and examples on how to use OpenNMT-tf, please visit [our documentation](https://opennmt.net/OpenNMT-tf).*

### Library

OpenNMT-tf also exposes [well-defined and stable APIs](https://opennmt.net/OpenNMT-tf/package/opennmt.html), from high-level training utilities to low-level model layers and dataset transformations.

For example, the `Runner` class can be used to train and evaluate models with few lines of code:

```python
import opennmt

config = {
    "model_dir": "/data/wmt-ende/checkpoints/",
    "data": {
        "source_vocabulary": "/data/wmt-ende/joint-vocab.txt",
        "target_vocabulary": "/data/wmt-ende/joint-vocab.txt",
        "train_features_file": "/data/wmt-ende/train.en",
        "train_labels_file": "/data/wmt-ende/train.de",
        "eval_features_file": "/data/wmt-ende/valid.en",
        "eval_labels_file": "/data/wmt-ende/valid.de",
    }
}

model = opennmt.models.TransformerBase()
runner = opennmt.Runner(model, config, auto_config=True)
runner.train(num_devices=2, with_eval=True)
```

Here is another example using OpenNMT-tf to run efficient beam search with a self-attentional decoder:

```python
decoder = opennmt.decoders.SelfAttentionDecoder(num_layers=6, vocab_size=32000)

initial_state = decoder.initial_state(
    memory=memory, memory_sequence_length=memory_sequence_length
)

batch_size = tf.shape(memory)[0]
start_ids = tf.fill([batch_size], opennmt.START_OF_SENTENCE_ID)

decoding_result = decoder.dynamic_decode(
    target_embedding,
    start_ids=start_ids,
    initial_state=initial_state,
    decoding_strategy=opennmt.utils.BeamSearch(4),
)
```

More examples using OpenNMT-tf as a library can be found online:

* The directory [examples/library](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/library) contains additional examples that use OpenNMT-tf as a library
* [nmt-wizard-docker](https://github.com/OpenNMT/nmt-wizard-docker) uses the high-level `opennmt.Runner` API to wrap OpenNMT-tf with a custom interface for training, translating, and serving

*For a complete overview of the APIs, see the [package documentation](https://opennmt.net/OpenNMT-tf/package/opennmt.html).*

## Additional resources

* [Documentation](https://opennmt.net/OpenNMT-tf)
* [Forum](https://forum.opennmt.net)
* [Gitter](https://gitter.im/OpenNMT/OpenNMT-tf)
