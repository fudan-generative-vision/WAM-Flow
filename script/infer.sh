#!/bin/bash

CKPT_PATH="pretrained_model/wam-flow/navsim"
FUDOKI_PATH="pretrained_model/fudoki"
IMAGE_PATH="data/navsim_data/sensor_blobs/test/2021.09.09.17.18.51_veh-48_00889_01147/CAM_F0/9a6f0331d98258a0.jpg"

torchrun --nproc_per_node 1 infer.py \
    --checkpoint_path $CKPT_PATH \
    --image_path $IMAGE_PATH \
    --processor_path $FUDOKI_PATH \
    --text_embedding_path $FUDOKI_PATH/text_embedding.pt \
    --image_embedding_path $FUDOKI_PATH/image_embedding.pt \
    --discrete_fm_steps 2 \
    --seed 123