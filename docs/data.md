# Data

Data files are configured in the `data` block of the YAML configuration file (see [Parameters](configuration.md)). For example, here is a typical data configuration for a sequence to sequence model:

```yaml
data:
  train_features_file: /data/ende/train.en
  train_labels_file: /data/ende/train.de
  eval_features_file: /data/ende/valid.en
  eval_labels_file: /data/ende/valid.en
```

This page documents how to prepare and configure data files for OpenNMT-tf.

## Terminology

* *features*: the model inputs (a.k.a. *source*)
* *inputters*: the input layer of the model that defines how to read element and transform them into vectorized inputs
* *labels*: the model outputs (a.k.a. *target*)

## File format

The format of the data files is defined by the `opennmt.inputters.Inputter` modules used by your model. OpenNMT-tf currently supports texts and sequence vectors as inputs.

### Text

All `opennmt.inputters.TextInputter` inputters expect a text file as input where:

* sentences are separated by a **newline**
* tokens are separated by a **space** (unless a custom tokenizer is set, see [Tokenization](tokenization.md))

For example:

```text
$ head -5 data/toy-ende/src-train.txt
It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
Federal Master Trainer and Senior Instructor of the Italian Federation of Aerobic Fitness , Group Fitness , Postural Gym , Stretching and Pilates; from 2004 , he has been collaborating with Antiche Terme as personal Trainer and Instructor of Stretching , Pilates and Postural Gym .
&quot; Two soldiers came up to me and told me that if I refuse to sleep with them , they will kill me . They beat me and ripped my clothes .
Yes , we also say that the European budget is not about the duplication of national budgets , but about delivering common goals beyond the capacity of nation states where European funds can realise economies of scale or create synergies .
The name of this site , and program name Title purchased will not be displayed .
```

### Sequence vector

The `opennmt.inputters.SequenceRecordInputter` expects a file with serialized [*TFRecords*](https://www.tensorflow.org/tutorials/load_data/tfrecord). We propose 2 ways to create this file, choose the one that is the easiest for you:

#### via Python

It is very simple to generate a compatible *TFRecords* file directly from Python:

```python
import opennmt as onmt
import numpy as np

dataset = [
  np.random.rand(8, 50),
  np.random.rand(4, 50),
  np.random.rand(13, 50)
]

onmt.inputters.create_sequence_records(dataset, "data.records")
```

This example saves a dataset of 3 random vectors of shape `[time, dim]` into the file `data.records`. It should be easy to adapt for any dataset of 2D vectors.

#### via the ARK text format

The script `onmt-ark-to-records` proposes an alternative way to generate this dataset. It converts the ARK text format:

```text
KEY [
FEAT1.1 FEAT1.2 FEAT1.3 ... FEAT1.n
...
FEATm.1 FEATm.2 FEATm.3 ... FEATm.n ]
```

which describes an example of `m` vectors of depth `n` and identified by `KEY`.

See `onmt-ark-to-records -h` for the script usage. It also accepts an optional indexed text file (i.e. with lines prefixed with `KEY`s) to generate aligned source vectors and target texts.

## Parallel inputs

When using `opennmt.inputters.ParallelInputter`, as many input files as inputters are expected. You have to configure your YAML file accordingly:

```yaml
data:
  train_features_file:
    - train_source_1.records
    - train_source_2.txt
    - train_source_3.txt
```

Similarly, when using the `--features_file` command line option of the main script (e.g. for inference or scoring), a list of files must also be provided:

```bash
onmt.main [...] infer \
    --features_file test_source_1.records test_source_2.txt test_source_3.txt
```

## Weighted inputs

It is possible to configure multiple training data files and assign weights to each file:

```yaml
data:
  train_features_file:
    - domain_a.en
    - domain_b.en
    - domain_c.en
  train_labels_file:
    - domain_a.de
    - domain_b.de
    - domain_c.de
  train_files_weights:
    - 0.5
    - 0.2
    - 0.3
```

This configuration will create a weighted dataset where examples will be randomly sampled from the data files according to the provided weights. The weights are normalized by the file size so that examples from small files are not repeated more often than examples from large files during the training.

## Compressed data

Data files compressed with GZIP are supported. The path should end with the `.gz` extension for the file to be correctly loaded:

```yaml
data:
  train_features_file: /data/wmt/train.en.gz
  train_labels_file: /data/wmt/train.de.gz
```

## Data location

By default, the data are expected to be on the same filesystem. However, it is possible to reference data stored in HDFS, Amazon S3, or any other remote storages supported by TensorFlow. For example:

```yaml
data:
  train_features_file: s3://namebucket/toy-ende/src-train.txt
  train_labels_file: hdfs://namenode:8020/toy-ende/tgt-train.txt
```

For more information, see the TensorFlow documentation:

* [How to run TensorFlow on Hadoop](https://www.tensorflow.org/deploy/hadoop)
* [How to run TensorFlow on S3](https://www.tensorflow.org/deploy/s3)
