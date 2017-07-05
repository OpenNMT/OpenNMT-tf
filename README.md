# OpenNMT-tf

## Requirements

* `tensorflow` (>= 1.2.1)
* `pyyaml`

## Usage

### Prepare vocabularies

See `python build_vocab.py -h` to generate vocabularies as needed by your model.

### Build your model

The user should provide a `opennmt.models.Model` instance using available or user-defined modules.

See example models in `config/models/`.

### Describe your run

Each run should have its own YAML file describing the run:

* type (training or inference)
* checkpoints directory
* batch size
* data
* etc.

See example configuration files in `config/`.

### Train or infer

```
python onmt.py --model config/models/my_model.py --run config/my_run.yml
```

### Monitor

Start [TensorBoard](https://github.com/tensorflow/tensorboard):

```
tensorboard --logdir="."
```

then open the URL displayed in the shell.

You will be able to monitor and visualize:

* training and evaluation loss
* training speed
* gradients norm
* computation graphs
* word embeddings
