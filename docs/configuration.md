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

Parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference.

*See the example configuration `config/sample.yml` to learn about available parameters.*

### Multiple configuration files

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```bash
onmt-main [...] --config config/opennmt-defaults.yml config/optim/adam_with_decay.yml config/data/toy-ende.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

If you are unsure about the configuration that is actually used or simply prefer working with a single file, consider using the `merge_config` script:

```bash
onmt-merge-config config/opennmt-defaults.yml config/optim/adam_with_decay.yml config/data/toy-ende.yml > config/my_config.yml
```
