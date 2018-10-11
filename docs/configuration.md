# Configuration

## Model

### Definition

Models are defined from the code to allow a high level of modeling freedom. They are `opennmt.models.Model` instances that use [available](package/opennmt.html) or user-defined modules. Some of these modules are defined to contain other modules and can be used to design complex architectures:

* `opennmt.encoders.ParallelEncoder`
* `opennmt.encoders.SequentialEncoder`
* `opennmt.inputters.MixedInputter`
* `opennmt.inputters.ParallelInputter`

For example, these container modules can be used to implement multi source inputs, multi modal training, mixed word/character embeddings, and arbitrarily complex encoder architectures (e.g. mixing convolution, RNN, self-attention, etc.).

### Usage

The user can either:

* select a predefined model from the [catalog](package/opennmt.models.catalog.html) and use the `--model_type` command line option
* **or** provide a custom configuration file that follows the template file `config/models/template.py` and use the `--model` command line option

*See the predefined models definitions in the [catalog](_modules/opennmt/models/catalog.html). More experimental models and examples are also available in the `config/models/` directory. Contributions to add more model definitions are welcome.*

## Parameters

Parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference. It uses the following layout:

```yaml
model_dir: path_to_the_model_directory

data:
  # Data configuration (training and evaluation files, vocabularies, alignments, etc.)
params:
  # Training and inference hyperparameters (learning rate, optimizer, beam size, etc.)
train:
  # Training specific configuration (checkpoint frequency, number of training step, etc.)
eval:
  # Evaluation specific configuration (evaluation frequency, external evaluators.)
infer:
  # Inference specific configuration (output scores, alignments, etc.)
score:
  # Scoring specific configuration
```

For a complete list of available options, see [Reference: Configuration](configuration_reference.html).

### Automatic configuration

Predefined models declare default parameters that should give solid performance out of the box. To enable automatic configuration, use the `--auto_config` flag:

```bash
onmt-main train_and_eval --model_type Transformer --config my_data.yml --auto_config
```

The user provided `my_data.yml` file will minimaly require the data configuration. You might want to also configure checkpoint related settings, the logging frequency, and the number of training steps.

At the start of the training, the configuration values actually used will be logged. If you want to change some of them, simply add the parameter in your configuration file to override the default value.

**Note:** default training values usually assume GPUs with at least 8GB of memory and a large system memory:

* If you encounter GPU out of memory issues, try overriding `batch_size` to a lower value.
* If you encounter CPU out of memory issues, try overriding `sample_buffer_size` to a fix value.

### Multiple configuration files

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```bash
onmt-main [...] --config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

If you are unsure about the configuration that is actually used or simply prefer working with a single file, consider using the `merge_config` script:

```bash
onmt-merge-config config/opennmt-defaults.yml config/optim/adam_with_decay.yml \
    config/data/toy-ende.yml > config/my_config.yml
```

## TensorFlow session

The command line option `--session_config` can be used to configure the TensorFlow session that is created to execute TensorFlow graphs. The option takes a file containing a [`tf.ConfigProto`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto) message serialized in text format.

Here is an example to enable the `allow_growth` GPU option:

```text
$ cat config/session_config.txt
gpu_options {
  allow_growth: true
}
```

```bash
onmt-main [...] --session_config config/session_config.txt
```

For possible options and values, see the [`tf.ConfigProto`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/protobuf/config.proto) file.
