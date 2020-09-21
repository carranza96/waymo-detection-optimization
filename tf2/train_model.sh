#!/bin/bash

MODEL_NAME=optimized_faster_rcnn
NUM_TRAIN_STEPS=50000
MODEL_DIR=saved_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config
GPU_DEVICE=0

python model_main_tf2.py \
    --model_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --gpu_device=${GPU_DEVICE} \





