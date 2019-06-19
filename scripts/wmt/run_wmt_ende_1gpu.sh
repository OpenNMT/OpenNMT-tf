#!/bin/bash

onmt-main \
          --model_type Transformer \
          --config config/wmt_ende.yml --auto_config train_and_eval
