# Configuration

## Model

Models are defined from the code to allow a high level of modeling freedom. The user should provide a `opennmt.models.Model` instance using [available](package/opennmt.html) or user-defined modules.

Some modules are defined to contain other modules and can be used to design complex architectures:

* `opennmt.encoders.ParallelEncoder`
* `opennmt.encoders.SequentialEncoder`
* `opennmt.inputters.MixedInputter`
* `opennmt.inputters.ParallelInputter`

For example, these container modules can be used to implement multi source inputs, multi modal training, mixed word/character embeddings, and arbitrarily complex encoder architectures (e.g. mixing convolution, RNN, self-attention, etc.).

*See the template file `config/models/template.py` and predefined models in `config/models/`. Contributions to add more model configurations are welcome.*

## Parameters

Parameters are described in separate YAML files. They define data files, optimization settings, dynamic model parameters, and options related to training and inference.

*See the example configuration `config/sample.yml` to learn about available parameters.*

### Multiple configuration files

The command line accepts multiple configuration files so that some parts can be made reusable, e.g:

```bash
python -m bin.main [...] --config config/opennmt-defaults.yml config/optim/adam_with_decay.yml config/data/toy-ende.yml
```

If a configuration key is duplicated, the value defined in the rightmost configuration file has priority.

If you are unsure about the configuration that is actually used or simply prefer working with a single file, consider using the `merge_config` script:

```bash
python -m bin.merge_config config/opennmt-defaults.yml config/optim/adam_with_decay.yml config/data/toy-ende.yml > config/my_config.yml
```
