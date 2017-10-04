## Serving

This example implements a simple client that sends a batch translation request to a model server.

### Requirements

* [TensorFlow Serving](https://www.tensorflow.org/serving) (`1.3.0`)
* an exported translation model

Note: for this example to work, the model should be a `SequenceToSequence` model with `WordEmbedder` as the source inputter.

### Usage

1\. Start `tensorflow_model_server`:

```
tensorflow_model_server --port=9000 --model_name=enfr --model_base_path=$HOME/OpenNMT-tf/models/enfr
```

* `enfr` is the model name that the client will request
* `$HOME/OpenNMT-tf/models/enfr` is a directory containing multiple versions of exported models (e.g. `v1` and `v2`)

2\. Run the client:

```
$ python examples/serving/nmt_client.py --host=localhost --port=9000 --model_name=enfr
Input:
	Hello world !
	My name is John .
	I live on the West coast .
Sending request...
Output:
	Bonjour le monde !
	Mon nom est John .
	Je vis sur la côte Ouest .
```
