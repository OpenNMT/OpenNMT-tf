# Serving

## Exporting a SavedModel

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

### Manual export

Models can be exported using the `export` run type:

```bash
onmt-main --config my_config.yml --auto_config export --export_dir ~/my-models/ende
```

It's a good idea to re-export your model with a more recent version as it can improve performance.

## Running a SavedModel

When using an exported model, you need to know the input and output nodes of your model. You can use the [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) script provided by TensorFlow for inspection, e.g.:

```bash
saved_model_cli show --dir ~/my-models/ende \
    --tag_set serve --signature_def serving_default
```

Some examples using exported models are available in the [`examples/serving`](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) directory.

## Input preprocessing and tokenization

TensorFlow Serving only runs TensorFlow operations. Preprocessing functions such as the tokenization is sometimes not implemented in terms of TensorFlow ops (see the *Tokenization* page for more details). In this case, these functions should be run outside of the TensorFlow engine, either by the client or a proxy server.

* The OpenNMT-tf [serving example](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) uses the client approach to implement a simple interactive translation loop
* The project [nmt-wizard-docker](https://github.com/OpenNMT/nmt-wizard-docker) uses the proxy server approach to wrap a TensorFlow Serving instance with a custom processing layer and REST API. Exported OpenNMT-tf models can integrated with this tool by following these [instructions](https://github.com/OpenNMT/nmt-wizard-docker/issues/46#issuecomment-456795844).
