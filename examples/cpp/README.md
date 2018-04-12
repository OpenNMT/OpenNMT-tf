## Inference with the TensorFlow C++ API

This example shows how to translate a batch of sentences using the TensorFlow C++ API.

### Requirements

* TensorFlow compiled as a library (see below)
* an exported translation model

**Note:** for this example to work, the model should be a `SequenceToSequence` model with `WordEmbedder` as the source inputter.

### Compile TensorFlow as a shared library

1\. [Prepare the environment](https://www.tensorflow.org/install/install_sources#prepare_environment_for_linux) for compiling TensorFlow.

2\. Download the TensorFlow repository:

```
git clone https://github.com/tensorflow/tensorflow.git ~/tensorflow
cd ~/tensorflow
git checkout r1.6
```

3\. Add a custom compilation target at the end of `tensorflow/BUILD`:


```
tf_cc_shared_object(
    name = "libtensorflow_opennmt.so",
    linkopts = select({
        "//tensorflow:darwin": [
            "-Wl,-exported_symbols_list",  # This line must be directly followed by the exported_symbols.lds file
            "//tensorflow:tf_exported_symbols.lds",
        ],
        "//tensorflow:windows": [],
        "//tensorflow:windows_msvc": [],
        "//conditions:default": [
            "-z defs",
            "-s",
            "-Wl,--version-script",  #  This line must be directly followed by the version_script.lds file
            "//tensorflow:tf_version_script.lds",
        ],
    }),
    deps = [
        "//tensorflow:tf_exported_symbols.lds",
        "//tensorflow:tf_version_script.lds",
        "//tensorflow/c:c_api",
        "//tensorflow/c/eager:c_api",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/core:tensorflow",
        "//tensorflow/contrib/seq2seq:beam_search_ops_kernels",
        "//tensorflow/contrib/seq2seq:beam_search_ops_op_lib",
    ],
)
```

4\. Configure and compile:

```
./configure
bazel build --config=opt //tensorflow:libtensorflow_opennmt.so
```

**Note:** this compilation approach is presented for the sole purpose of this example. It may not be the recommended way to work with the TensorFlow C++ API.

### Usage

```
export LD_LIBRARY_PATH=$HOME/tensorflow/bazel-bin/tensorflow
make
./main --export_dir=$HOME/OpenNMT-tf/models/enfr/1507109306/
```

where `$HOME/OpenNMT-tf/models/enfr/1507109306/` is a directory containing the exported model.

The output of the last command should look like this:

```
2017-10-04 18:20:22.190490: I tensorflow/cc/saved_model/loader.cc:236] Loading SavedModel from: $HOME/OpenNMT-tf/models/enfr/1507109306/
2017-10-04 18:20:22.228595: I tensorflow/cc/saved_model/loader.cc:155] Restoring SavedModel bundle.
2017-10-04 18:20:22.373715: I tensorflow/cc/saved_model/loader.cc:190] Running LegacyInitOp on SavedModel bundle.
2017-10-04 18:20:22.407657: I tensorflow/cc/saved_model/loader.cc:284] Loading SavedModel: success. Took 217175 microseconds.
Input:
 Hello world !
 My name is John .
 I live on the West coast .
Output:
 Bonjour le monde !
 Mon nom est John .
 Je vis sur la côte Ouest .
```
