# Inference with TensorFlow Serving

This example shows how to start a TensorFlow Serving GPU instance and sends translation requests via a simple Python client.

## Requirements

* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## Usage

**1\. Go into this directory, as assumed by the rest of the commands:**

```bash
cd examples/serving
```

**2\. Download the English-German pretrained model:**

```bash
wget https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz
tar xf averaged-ende-export500k.tar.gz
mv averaged-ende-export500k ende
```

**3\. Start a TensorFlow Serving GPU instance in the background:**

```bash
nvidia-docker run -d --rm -p 9000:9000 -v $PWD:/models \
  --name tensorflow_serving --entrypoint tensorflow_model_server \
  tensorflow/serving:1.11.0-gpu \
  --enable_batching=true --batching_parameters_file=/models/batching_parameters.txt \
  --port=9000 --model_base_path=/models/ende --model_name=ende
```

*For more information about the `batching_parameters.txt` file, see the [TensorFlow Serving Batching Guide](https://github.com/tensorflow/serving/tree/master/tensorflow_serving/batching).*

**4\. Install the client dependencies:**

```bash
pip install -r requirements.txt
```

**5\. Run the client:**

```bash
python ende_client.py --port 9000 --model_name ende \
  --sentencepiece_model ende/1539080952/assets.extra/wmtende.model
```

The output of this command should look like this (the model and client initialization might take some time):

```text
Hello world! ||| Hallo Welt!
My name is John. ||| Mein Name ist John.
I live on the West coast. ||| Ich lebe an der Westküste.
```

**6\. Stop TensorFlow Serving:**

```bash
docker kill tensorflow_serving
```

## Going further

Depending on your production requirements, you might need to build a simple proxy server between your application and TensorFlow Serving in order to:

* manage multiple TensorFlow Serving instances (possibly running on multiple hosts) and keep persistent channels to them
* apply tokenization and detokenization

For example, take a look at the OpenNMT-tf integration in the project [`nmt-wizard-docker`](https://github.com/OpenNMT/nmt-wizard-docker/blob/master/frameworks/opennmt_tf/entrypoint.py) which wraps a TensorFlow serving instance with a custom processing layer and REST API.
