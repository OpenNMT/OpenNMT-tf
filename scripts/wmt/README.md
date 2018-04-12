# Steps to train a Transformer model on the WMT English-German dataset

## Requirements

* OpenNMT-tf (>= 1.1.0)
* SentencePiece

Please follow the instructions to install and build [SentencePiece](https://github.com/google/sentencepiece). If installed in a custom location, change the `SP_PATH` variable in the scripts.

```bash
pip install OpenNMT-tf[tensorflow_gpu]>=1.1.0
```

## Steps

### Data preparation

Before running the script, look at the links in the file header to download the datasets. Depending on the task, you may need to change the filenames and the folders paths.

```bash
./prepare_data.sh /data/wmt/
```

where `/data/wmt/` contains the raw parallel datasets.

The script will train a SentencePiece model using the same source and target vocabulary. It will tokenize the dataset and prepare the train/valid/test files. A new directory `data/` will contain the generated files.

### Training

We recommend training on 4 GPUs to get the best performance:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_wmt_ende.sh
```

Or if you have only 1 GPU, run the dedicated script:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_wmt_ende_1gpu.sh
```

### Translate

```bash
./eval_wmt_ende.sh /data/wmt/
```

## Lazy run...

* [SentencePiece model and vocabulary](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp_model.tar.gz)
* [Pre-tokenized dataset](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz)
* Pre-trained averaged model:
  * [checkpoint](https://s3.amazonaws.com/opennmt-models/averaged-ende-ckpt500k.tar.gz)
  * [export](https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz)

This model achieved the following scores:

| Test set | NIST BLEU |
| --- | --- |
| newstest2014 | 26.9 |
| neswtest2017 | 28.0 |
