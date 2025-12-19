#!/bin/bash

NUM_NODES=1
NUM_GPUS=1

config=config/debug.yaml
output_dir=output/train/debug

accelerate launch \
  --config_file ./config/accelerate_config_ds2.yaml \
  --machine_rank 0 \
  --main_process_port 12345 \
  --num_machines $NUM_NODES \
  --num_processes $NUM_GPUS \
  train.py \
  --config $config \
  --output_dir $output_dir
