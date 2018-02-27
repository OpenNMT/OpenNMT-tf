
# Steps to train a transformer model based on WMT EN-DE

# Requirements

* `tensorflow` (`1.5`)
* `pyyaml`
* `SentencePiece`

Please follow instructions to install and build Google SentencePiece on this page: https://github.com/google/sentencepiece

Once it's installed, do not forget to change the SP_PATH variable in scripts.

## Data Preparation

Before running the script, look at the links to download the Datasets.
depending on the task, you may change the filenames and the Folders PATH.

cd scripts/wmt
./prepare_data.sh

The script will train a SentencePiece Model using the same source and target vocab.
It will tokenize the Dataset and prepare the train/valid/test files.

## Training

cd scripts/wmt
./run_wmt_ende.sh

By default (to be modified in wmt-ende.yml) training will be done on 4 GPUs and during 200 000 steps.


## Translate

cd scripts/wmt
./eval_wmt_ende.sh

## Lazy run .....

You can download directly the pre-tokenized SP dataset here: https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz
you can download the pre-trained averaged model here: https://s3.amazonaws.com/opennmt-trainingdata/averaged-ende-ckpt500k.tar.gz

Nist Bleu on Newstest 2014: 26.9
Nist Bleu on Neswtest 2017: 28.0

