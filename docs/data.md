# Data

## Data format

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

### Vectors

The `opennmt.inputters.SequenceRecordInputter` expects a file with serialized *TFRecords*. To simplify the preparation of these data, the script `onmt-ark-to-records` can be used to convert vectors serialized in the ARK text format:

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
