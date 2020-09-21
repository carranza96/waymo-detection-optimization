#!/bin/bash

MODEL_NAME=optimized_faster_rcnn
SAMPLE_1_OF_N_EVAL_EXAMPLES=1000
GPU_DEVICE=0


MODEL_DIR=saved_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config

  python model_main_tf2.py \
    --model_dir=${MODEL_DIR} \
    --checkpoint_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --eval_timeout=0 \
    --gpu_device=${GPU_DEVICE} \





