#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHONPATH=../.. python -m bin.main train_and_eval \
  --model ../../config/models/transformer.py \
  --config config/wmt_ende.yml \
  --num_gpus 4
