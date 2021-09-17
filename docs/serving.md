# Serving

## TensorFlow

### Exporting a SavedModel

OpenNMT-tf can export [SavedModel](https://www.tensorflow.org/guide/saved_model) packages for inference in other environments, for example with [TensorFlow Serving](https://www.tensorflow.org/serving/). A model export contains all information required for inference: the graph definition, the weights, and external assets such as vocabulary files. It typically looks like this on disk:

```text
toy-ende/export/
├── assets
│   ├── src-vocab.txt
│   └── tgt-vocab.txt
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

Models can be manually exported using the `export` run type:

```bash
onmt-main --config my_config.yml --auto_config export --output_dir ~/my-models/ende
```

Automatic evaluation during the training can also export models, see [Training](training.md) to learn more.

### Running a SavedModel

Once a SavedModel is exported, OpenNMT-tf is no longer needed to run it. However, you will need to know the input and output nodes of your model. You can use the [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) script provided by TensorFlow for inspection, e.g.:

```bash
saved_model_cli show --dir ~/my-models/ende \
    --tag_set serve --signature_def serving_default
```

Some examples using exported models are available in the [`examples/serving`](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) directory.

### Input preprocessing and tokenization

TensorFlow Serving only runs TensorFlow operations. Preprocessing functions such as the tokenization is sometimes not implemented in terms of TensorFlow ops (see [Tokenization](tokenization.md) for more details). In this case, these functions should be run outside of the TensorFlow runtime, either by the client or a proxy server.

## CTranslate2

[CTranslate2](https://github.com/OpenNMT/CTranslate2) is an optimized inference engine for OpenNMT models that is typically faster, lighter, and more customizable than the TensorFlow runtime.

Selected models can be exported to the CTranslate2 format directly from OpenNMT-tf by selecting the `ctranslate2` export format.

**When using the `export` command line**, the `--format` option should be set:

```bash
onmt-main [...] export --output_dir ~/my-models/ende --format ctranslate2
```

**When using the automatic model evaluation and export during the training**, the `export_format` option should be configured in the `eval` block of the YAML configuration:

```yaml
eval:
  scorers: bleu
  export_on_best: bleu
  export_format: ctranslate2
```

[Model quantization](https://github.com/OpenNMT/CTranslate2#quantization-and-reduced-precision) can also be enabled by replacing `ctranslate2` by one of the following export types:

* `ctranslate2_int8`
* `ctranslate2_int16`
* `ctranslate2_float16`


## TensorFlow Lite

[TensorFlow Lite](https://www.tensorflow.org/lite) is a deep learning framework for fast inference of TensorFlow models on mobile devices.

Converting to TensorFlow Lite requires **TensorFlow version 2.5+**

Example export command:
```bash
onmt-main [...] export --export_dir ~/output --export_format tflite 
```

Exporting will create an `opennmt.tflite` model file in the export directory.

**Compatible models**
* RNN models
* Transformer
* Transformer Relative
* Transformer Shared Embeddings

[Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization) is a way to decrease the model size, and the inference time.  

**Quantization Export Formats**
* Dynamic range quantization - `tflite_dynamic_range`
* Float16 quantization - `tflite_float16`

**Running a TFLite Model**

Running requires using the same vocabulary files used for training.

1. Convert the sentence to IDs with the vocabulary file.
2. Run the model with the IDs to get a fixed size array. ([TensorFlow Guide](https://www.tensorflow.org/lite/guide/inference))
3. Convert the resulting IDs to a sentence using the other vocabulary file.

**Model Output**

The model outputs a fixed size array, this can be specified in the data configuration file.
```yaml
params:
  tflite_output_size: 250
```

