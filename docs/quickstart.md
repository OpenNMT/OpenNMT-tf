# Quickstart

This page presents a minimal workflow to get you started in using OpenNMT-tf.

## Step 0: Install OpenNMT-tf

We recommend using [`virtualenv`](https://virtualenv.pypa.io/en/stable/) to setup and configure the environment for this quickstart:

```bash
virtualenv pyenv
source pyenv/bin/activate
pip install OpenNMT-tf[tensorflow_gpu]
```

## Step 1: Prepare the data

To get started, we propose to download a toy English-German dataset for machine translation containing 10k tokenized sentences:

```bash
wget https://s3.amazonaws.com/opennmt-trainingdata/toy-ende.tar.gz
tar xf toy-ende.tar.gz
cd toy-ende
```

The first step is to build the source and target **word vocabularies** from the training files:

```bash
onmt-build-vocab --size 50000 --save_vocab src-vocab.txt src-train.txt
onmt-build-vocab --size 50000 --save_vocab tgt-vocab.txt tgt-train.txt
```

Then, the data files should be declared in a **YAML configuration file**, let's name it `data.yml`:

```yaml
model_dir: run/

data:
  train_features_file: src-train.txt
  train_labels_file: tgt-train.txt
  eval_features_file: src-val.txt
  eval_labels_file: tgt-val.txt
  source_words_vocabulary: src-vocab.txt
  target_words_vocabulary: tgt-vocab.txt
```

## Step 2: Train the model

```
onmt-main train_and_eval --model_type NMTSmall --auto_config --config data.yml
```

This command will start the training and evaluation loop of a small RNN-based sequence to sequence model. The `--auto_config` flag selects the best settings for this type of model.

The training will regularly produce checkpoints in the `run/` directory. To monitor the training progress, some logs are displayed in the console. However, to visually monitor the training we suggest using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard):

```bash
tensorboard --logdir="run"
```

## Step 3: Translate

```
onmt-main infer --auto_config --config data.yml --features_file src-test.txt
```

This command can be executed as soon as a checkpoint is saved by the training; the most recent checkpoint will be used by default. The predictions will be printed on the standard output.

**That's it! You successfully executed the 3 main steps to prepare, run, and evaluate an OpenNMT-tf training.**

## Going further

While this example gave you a quick overview of a typical OpenNMT-tf workflow, it will not produce state of the art results. The selected dataset and model are too small for this task.

To go further, here are some pointers:

* Download larger training sets, for example from a [WMT task](http://www.statmt.org/wmt16/translation-task.html)
* Train a bigger model, like the [Transformer](https://arxiv.org/abs/1706.03762) using `--model_type Transformer`
* Run existing [training recipes](https://github.com/OpenNMT/OpenNMT-tf/tree/master/scripts)
* Discover the [configuration reference](configuration.html) to tune hyperparameters
* Explore the other sections to learn about advanced topics
