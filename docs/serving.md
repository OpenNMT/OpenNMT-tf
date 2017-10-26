# Serving

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

Models are automatically exported during the training or manually with the `export` run type.

When using an exported model, you need to know the input and output nodes of your model. You can use the [`saved_model_cli`](https://www.tensorflow.org/programmers_guide/saved_model#cli_to_inspect_and_execute_savedmodel) script provided by TensorFlow for inspection, e.g.:

```bash
saved_model_cli show --dir toy-ende/export/latest/1507109306/ --tag_set serve --signature_def serving_default
```

Some examples using exported models are available in the `examples/` directory:

* `examples/serving` to serve a model with TensorFlow Serving
* `examples/cpp` to run inference with the TensorFlow C++ API

**Note:** because the Python function used in `tf.py_func` is not serialized in the graph, model exports do not support in-graph tokenization and text inputs are expected to be tokenized.
