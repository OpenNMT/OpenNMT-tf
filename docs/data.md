# Data

## File format

The format of the data files is defined by the `opennmt.inputters.Inputter` used by your model.

### Text

All `opennmt.inputters.TextInputter`s expect a text file as input where:

* sentences are separated by a **newline**
* tokens are separated by a **space** (unless a custom tokenizer is set, see [Tokenization](tokenization.html))

For example:

```text
$ head -5 data/toy-ende/src-train.txt
It is not acceptable that , with the help of the national bureaucracies , Parliament &apos;s legislative prerogative should be made null and void by means of implementing provisions whose content , purpose and extent are not laid down in advance .
Federal Master Trainer and Senior Instructor of the Italian Federation of Aerobic Fitness , Group Fitness , Postural Gym , Stretching and Pilates; from 2004 , he has been collaborating with Antiche Terme as personal Trainer and Instructor of Stretching , Pilates and Postural Gym .
&quot; Two soldiers came up to me and told me that if I refuse to sleep with them , they will kill me . They beat me and ripped my clothes .
Yes , we also say that the European budget is not about the duplication of national budgets , but about delivering common goals beyond the capacity of nation states where European funds can realise economies of scale or create synergies .
The name of this site , and program name Title purchased will not be displayed .
```

### Vocabulary

For text inputs, vocabulary files should be provided in the data configuration (see for example in the *Quickstart* section). OpenNMT-tf uses a simple text format with **one token per line**, which should begin with these special tokens:

```text
<blank>
<s>
</s>
```

The `onmt-build-vocab` script can be used to generate this file.

#### from tokenized files

The script can be run directly on tokenized files. For example the command:

```bash
onmt-build-vocab --size 50000 --save_vocab vocab.txt train.txt.tok
```

extracts the 50,000 most frequent tokens from `train.txt.tok` and saves them to `vocab.txt`.

#### from raw files

By default, `onmt-build-vocab` splits each line on spaces. It is possible to define a custom tokenization with the `--tokenizer_config` option. See the *Tokenization* section for more information.

#### from SentencePiece training

`onmt-build-vocab` can also prepare a SentencePiece vocabulary and model from raw data. For example the command:

```bash
onmt-build-vocab --sentencepiece --size 32000 --save_vocab sp train.txt.raw
```

will produce the SentencePiece model `sp.model` and the vocabulary `sp.vocab` of size 32,000. Additional SentencePiece [training options](https://github.com/google/sentencepiece/blob/master/src/spm_train_main.cc) can be passed to the `--sentencepiece` argument in the format `option=value`, e.g. `--sentencepiece character_coverage=0.98 num_threads=4`.

### Vectors

The `opennmt.inputters.SequenceRecordInputter` expects a file with serialized *TFRecords*. We propose 2 ways to create this file, choose the one that is the easiest for you:

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

This example saves a dataset of 3 random vectors of shape `[time, dim]` into the file "data.records". It should be easy to adapt for any dataset of 2D vectors.

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

### Alignments

Guided alignment requires a training alignment file that uses the "Pharaoh" format, e.g.:

```text
0-0 1-1 2-4 3-2 4-3 5-5 6-6
0-0 1-1 2-2 2-3 3-4 4-5
0-0 1-2 2-1 3-3 4-4 5-5
```

where a pair `i-j` indicates that the `i`th word of the source sentence is aligned with the `j`th word of the target sentence (zero-indexed).

This file should then be added in the data configuration:

```yaml
data:
  train_alignments: train-alignment.txt

params:
  guided_alignment_type: ce
```

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

## Parallel inputs

When using `opennmt.inputters.ParallelInputter`, as many input files as inputters are expected. You have to configure your YAML file accordingly:

```yaml
data:
  train_features_file:
    - train_source_1.records
    - train_source_2.txt
    - train_source_3.txt

  source_2_vocabulary: ...
  source_3_vocabulary: ...

  # If you also want to configure the tokenization:
  source_2_tokenization: ...
  source_3_tokenization: ...

  # If you also want to configure the embeddings:
  source_2_embedding: ...
  source_3_embedding: ...
```

Similarly, when using the `--features_file` command line option of the main script (e.g. for inference or scoring), a list of files must also be provided:

```bash
onmt.main [...] infer \
    --features_file test_source_1.records test_source_2.txt test_source_3.txt
```
