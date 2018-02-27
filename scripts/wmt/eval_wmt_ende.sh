#!/bin/bash
sl=en
tl=de

runconfig=wmt_ende_transformer_4gpu_lr2_ws8000_dur2_0.998
DATA_PATH=/mosesgpu4-hd1TB/datasets/de_DE-en_EN/public
testset=newstest2017-ende

export CUDA_VISIBLE_DEVICES=0
SP_PATH=~/sentencepiece/src

../../third_party/input-from-sgm.perl < $DATA_PATH/test/$testset-src.$sl.sgm \
   | $SP_PATH/spm_encode --model=data/wmt$sl$tl.model > data/$testset-src.$sl
../../third_party/input-from-sgm.perl < $DATA_PATH/test/$testset-ref.$tl.sgm > data/$testset-ref.$tl


if false; then
  mkdir -p $runconfig/averaged
PYTHONPATH=../.. python bin/average_checkpoints.py --max_count=10 \
  --model_dir=$runconfig/ --output_dir=$runconfig/averaged/
fi

if true; then
PYTHONPATH=../.. python -m bin.main infer --config wmt_ende.yml --checkpoint_path=$runconfig/averaged \
    --features_file data/$testset-src.$sl > data/$testset-src.hyp.$tl
fi
if true; then
  $SP_PATH/spm_decode --model=data/wmt$sl$tl.model --input_format=piece < data/$testset-src.hyp.$tl > data/$testset-src.hyp.detok.$tl

  ../../third_party/multi-bleu-detok.perl data/$testset-ref.$tl < data/$testset-src.hyp.detok.$tl
fi
