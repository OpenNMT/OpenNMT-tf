#!/bin/bash

if [ $# -eq 0 ]
then
    echo "usage: $0 <data_dir>"
    exit 1
fi

DATA_PATH=$1
SP_PATH=/usr/local/bin

runconfig=wmt_ende_transformer
testset=newstest2017-ende
sl=en
tl=de

export PATH=$SP_PATH:$PATH
export CUDA_VISIBLE_DEVICES=0

wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/input-from-sgm.perl
wget -nc https://raw.githubusercontent.com/OpenNMT/OpenNMT-tf/master/third_party/multi-bleu-detok.perl

perl input-from-sgm.perl < $DATA_PATH/test/$testset-src.$sl.sgm \
   | spm_encode --model=data/wmt$sl$tl.model > data/$testset-src.$sl
perl input-from-sgm.perl < $DATA_PATH/test/$testset-ref.$tl.sgm > data/$testset-ref.$tl

if true; then
  onmt-main \
            --config config/wmt_ende.yml --auto_config \
            --checkpoint_path=$runconfig/avg \
            infer \
            --features_file data/$testset-src.$sl \
            > data/$testset-src.hyp.$tl
fi

if true; then
  spm_decode --model=data/wmt$sl$tl.model --input_format=piece \
             < data/$testset-src.hyp.$tl \
             > data/$testset-src.hyp.detok.$tl

  perl multi-bleu-detok.perl data/$testset-ref.$tl < data/$testset-src.hyp.detok.$tl
fi
