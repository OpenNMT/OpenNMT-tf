[![Build Status](https://api.travis-ci.org/OpenNMT/OpenNMT-tf.svg?branch=master)](https://travis-ci.org/OpenNMT/OpenNMT-tf) [![Documentation](https://img.shields.io/badge/docs-master-blue.svg)](http://opennmt.net/OpenNMT-tf/) [![Gitter](https://badges.gitter.im/OpenNMT/OpenNMT-tf.svg)](https://gitter.im/OpenNMT/OpenNMT-tf?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)

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

You can download directly the pre-tokenized SP dataset here: 
you can download the pre-trained averaged model here:

## Acknowledgments

The implementation is inspired by the following:

* [TensorFlow's NMT tutorial](https://github.com/tensorflow/nmt)
* [Tensor2Tensor](https://github.com/tensorflow/tensor2tensor)
* [Google's seq2seq](https://github.com/google/seq2seq)

## Additional resources

* [Documentation](http://opennmt.net/OpenNMT-tf)
* [Forum](http://forum.opennmt.net)
