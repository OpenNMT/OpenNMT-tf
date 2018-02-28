#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

PYTHONPATH=../.. python -m bin.main train --model ../../config/models/transformer.py \
  --config config/wmt_ende.yml config/adam_with_noam_decay.yml --num_gpus 4
