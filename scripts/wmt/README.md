# Steps to train a Transformer model on the WMT English-German dataset

This tutorial outlines the steps required to train a Transformer model (originally introduced [here](https://arxiv.org/abs/1706.03762)) on the WMT English-German dataset.

This model achieved the following scores:

| Test set | NIST BLEU |
| --- | --- |
| newstest2014 | 26.9 |
| newstest2017 | 28.0 |

## Requirements

+ OpenNMT-tf (>= 2.0.0)
+ SentencePiece (see installation instructions below)

## Steps

The following instructions are for replicating the result from scratch; skip to the bottom for a "Lazy run" using pre-calculated components.

### Installing SentencePiece

NMT models perform better if words are represented as sub-words, since this helps the out-of-vocabulary problem. [SentencePiece](https://arxiv.org/pdf/1808.06226.pdf) is a powerful end-to-end tokenizer that allows the learning of subword units from raw data. [We will install SentencePiece from source](https://github.com/google/sentencepiece#c-from-source) rather than via `pip install`, since the `spm_train` command used for training a SentencePiece model is not installed via pip but has to be built from the C++.

Installation instructions are available [here](https://github.com/google/sentencepiece#c-from-source). (For a TensorFlow Docker container built from latest-gpu-py3 image, running Ubuntu 16.04; TensorFlow Docker images are [here](https://hub.docker.com/r/tensorflow/tensorflow/), a beginner's tutorial on Docker and containerisation is [here](https://docker-curriculum.com/).)

If you installed SentencePiece in a custom location, change the `SP_PATH` variable in the scripts.

### Data loading and preparation with SentencePiece

The `prepare_data.sh` script automatically downloads the default datasets (`commoncrawl.de-en`, `europarl-v7.de-en`, `news-commentary-v11.de-en`, `newstest2014-deen` and `newstest2017-ende`) using wget, extracts the files, and tidies the folders for you. If you would like to change the task or datasets used, look at the links in the file header to find their download paths. Changing the source and target language is also possible as long as the associated datasets are available. Be sure to read the logic of the file cleaning to see what has to be adapted to new data.

The script will also train a SentencePiece model using the same source and target vocabulary. It will tokenize the dataset and prepare the train/valid/test files. The generated files will go into a `data/` directory.

Run these steps using the command:

```bash
./prepare_data.sh raw_data
```

where `raw_data` is the name of the folder that the raw parallel datasets will be downloaded into.


### Training the Transformer model

Now that the train/valid/test files have been generated, kick off a training run using the following command. We recommend training on 4 GPUs to get the best performance.


```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run_wmt_ende.sh
```

If you have e.g. 3 GPUs, change the `--num-gpus=4` in the `run_wmt_ende.sh` script to `--num-gpus=3` and set `CUDA_VISIBLE_DEVICES=0,1,2`.

Or if you have only 1 GPU, run the dedicated script:

```bash
CUDA_VISIBLE_DEVICES=0 ./run_wmt_ende_1gpu.sh
```

### Monitoring model training using TensorBoard

You can launch TensorBoard training monitoring by specifying where the logs are being written to (in this case, to a folder called `wmt_ende_transformer`). For example:

`tensorboard --logdir="/path/to/wmt_ende_transformer" --port=6006`

You can then open a browser and go to `<IP address>:6006` (e.g. http://127.0.0.1:6006/ if running on your local machine) to see TensorBoard graphs of gradients, loss, BLEU, learning rate, etc. over time as training proceeds. For an introduction to using TensorBoard, see this [video](https://www.youtube.com/watch?v=eBbEDRsCmv4) or this [post](https://www.datacamp.com/community/tutorials/tensorboard-tutorial).

If you are using NVIDIA GPUs, you can monitor card usage during training using `watch -n 0.5 nvidia-smi`.

If you are using a Docker container, you can launch a TensorBoard run from outside the container using:

`docker exec <CONTAINER ID> tensorboard --logdir="path/to/wmt_ende_transformer" --port=6006`

Note that for this to work, you need to have exposed the port when you create your Docker container ([simple example](https://briancaffey.github.io/2017/11/20/using-tensorflow-and-tensor-board-with-docker.html)).

### Translation using a trained model

You can run the following script to perform inference on the test set using a trained model. The script calls `onmt-main infer`. Normally, the latest checkpoint is used for inference by default, but [we recommend averaging the parameters of several checkpoints](http://opennmt.net/OpenNMT-tf/inference.html#checkpoints-averaging), which usually boosts model performance.

If training is left to run until completion, checkpoint averaging is automatically run. To average the last 5 checkpoints manually, the command is:

```bash
onmt-main --config config/wmt_ende.yml --auto_config average-checkpoints --output_dir wmt_ende_transformer/avg --max_count 5
```
And finally, to run inference:

```bash
./eval_wmt_ende.sh raw_data
```
You can then use your own evaluation scripts for e.g. computing BLEU or METEOR scores on the translated test set.

## Lazy run...

* [SentencePiece model and vocabulary](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp_model.tar.gz)
* [Pre-tokenized dataset](https://s3.amazonaws.com/opennmt-trainingdata/wmt_ende_sp.tar.gz)
* Pre-trained averaged model:
  * [checkpoint](https://s3.amazonaws.com/opennmt-models/averaged-ende-ckpt500k.tar.gz)
  * [export](https://s3.amazonaws.com/opennmt-models/averaged-ende-export500k.tar.gz)
