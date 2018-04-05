## Inference with TensorFlow Serving

This example implements a simple client that sends translation requests to a model server managed by TensorFlow Serving.

### Requirements

* [TensorFlow Serving](https://www.tensorflow.org/serving)
* an exported translation model

**Note:** for this example to work, the model should be a `SequenceToSequence` model with `WordEmbedder` as the source inputter.

### Usage

1\. Start `tensorflow_model_server`:

```
tensorflow_model_server --port=9000 --enable_batching=true --batching_parameters_file=examples/serving/batching_parameters.txt --model_name=enfr --model_base_path=$HOME/OpenNMT-tf/models/enfr
```

* `examples/serving/batching_parameters.txt` contains settings related to the batching mechanism
* `enfr` is the model name that the client will request
* `$HOME/OpenNMT-tf/models/enfr` is a directory containing multiple exported versions of the same model

2\. Run the client:

```
python examples/serving/nmt_client.py --host=localhost --port=9000 --model_name=enfr
```

The output of this command should look like this:

```
Hello world ! ||| Bonjour le monde !
My name is John . ||| Mon nom est John .
I live on the West coast . ||| Je vis sur la côte Ouest .
```

See the next section to understand why the last translation took more time to appear.

### Batching with TensorFlow Serving

In this example, the translation requests are sent separately because batching is handled on the server side. The `batching_parameters.txt` should be carefully tuned to achieve the throughput/latency balance that is suited for your application.

In particular, the provided configuration makes the server run a batch translation if:

* the number of queued requests has reached `max_batch_size` (here 2)
* **or** `batch_timeout_micros` microseconds has passed (here 5,000,000)

This explains why running the command above, the first 2 translations were received immediately (`max_batch_size` reached) and the third came 5 seconds later (`batch_timeout_micros` reached).

For more information, see the [TensorFlow Serving Batching Guide](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching).
