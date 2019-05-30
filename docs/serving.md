# Serving

## Exporting a SavedModel

OpenNMT-tf periodically exports models for inference in other environments, for example with [TensorFlow Serving](https://www.tensorflow.org/serving/). A model export contains all information required for inference: the graph definition, the weights, and external assets such as vocabulary files. It typically looks like this on disk:

```text
toy-ende/export/latest/1507109306/
├── assets
│   ├── src-vocab.txt
│   └── tgt-vocab.txt
├── saved_model.pb
└── variables
    ├── variables.data-00000-of-00001
    └── variables.index
```

### Automatic export

In the `train_and_eval` run type, models can be automatically exported following one or several export schedules:

* `last`: a model is exported to `export/latest` after each evaluation (default);
* `final`: a model is exported to `export/final` at the end of the training;
* `best`: a model is exported to `export/best` only if it achieves the best evaluation loss so far.

Export schedules are set by the `exporters` field in the `eval` section of the configuration file, e.g.:

```yaml
eval:
  exporters: best
```

### Manual export

Additionally, models can be manually exported using the `export` run type. This step is for example needed for changing some decoding configurations (e.g. beam size, minimum decoding length, etc.). Re-exporting your model with a more recent version can also improve its performance.

Manually exported models are located by default in `export/manual/` within the model directory; a custom destination can be configured with the command line option `--export_dir_base`, e.g.:

```bash
onmt-main export --export_dir_base ~/my-models/ende --auto_config --config my_config.yml
```

## Running a SavedModel

When using an exported model, you need to know the input and output nodes of your model. You can use the [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) script provided by TensorFlow for inspection, e.g.:

```bash
saved_model_cli show --dir toy-ende/export/latest/1507109306/ \
    --tag_set serve --signature_def serving_default
```

Some examples using exported models are available in the [`examples/`](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples) directory:

* `examples/serving` to serve a model with TensorFlow Serving
* `examples/cpp` to run inference with the TensorFlow C++ API

## Input preprocessing

TensorFlow Serving only runs TensorFlow operations. Preprocessing functions such as the tokenization is usually not implemented in terms of TensorFlow operators. These functions should be run outside of the TensorFlow engine, either by the client or a proxy server.

* The OpenNMT-tf [serving example](https://github.com/OpenNMT/OpenNMT-tf/tree/master/examples/serving) uses the client approach to implement a simple interactive translation loop
* The project [nmt-wizard-docker](https://github.com/OpenNMT/nmt-wizard-docker) uses the proxy server approach to wrap a TensorFlow Serving instance with a custom processing layer and REST API. Exported OpenNMT-tf models can integrated with this tool by following these [instructions](https://github.com/OpenNMT/nmt-wizard-docker/issues/46#issuecomment-456795844).
