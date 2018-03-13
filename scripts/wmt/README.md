# Steps to train a Transformer model on the WMT English-German dataset

## Requirements

* `tensorflow` (`1.5`)
* `pyyaml`
* `sentencepiece`

Please follow the instructions to install and build [SentencePiece](https://github.com/google/sentencepiece). Once it's installed, do not forget to change the `SP_PATH` variable in scripts.

## Data preparation

Before running the script, look at the links to download the datasets. Depending on the task, you may change the filenames and the folders paths.

```bash
cd scripts/wmt
./prepare_data.sh
```

The script will train a SentencePiece model using the same source and target vocabulary. It will tokenize the dataset and prepare the train/valid/test files.

## Training

```bash
cd scripts/wmt
./run_wmt_ende.sh
```

By default (to be modified in `wmt-ende.yml`) training will be done on 4 GPUs and during 200,000 steps.

## Translate

```bash
cd scripts/wmt
./eval_wmt_ende.sh
```

## Lazy run...

* [Pre-tokenized SentencePiece dataset](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz)
* Pre-trained averaged model:
  * [checkpoint](https://s3.amazonaws.com/opennmt-models/averaged-ende-ckpt500k.tar.gz)
  * [export](https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz)

This model achieved the following scores:

| Test set | NIST BLEU |
| --- | --- |
| newstest2014 | 26.9 |
| neswtest2017 | 28.0 |
