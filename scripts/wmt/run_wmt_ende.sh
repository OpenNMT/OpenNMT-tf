#!/bin/bash

onmt-main \
          --model_type Transformer \
          --config config/wmt_ende.yml --auto_config \
          --num_gpus 4 train_and_eval
