MODEL_NAME=optimized_faster_rcnn
GPU_DEVICE=0
NUM_TRAIN_STEPS=300000
SAMPLE_1_OF_N_EVAL_EXAMPLES=200
SAVE_CHECKPOINTS_SECS=1000
DEBUG_TENSORBOARD=False


MODEL_DIR=saved_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config

python src/model_main.py \
    --model_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --save_checkpoints_secs=${SAVE_CHECKPOINTS_SECS} \
    --num_gpu=${NUM_GPU} \
    --gpu_device=${GPU_DEVICE} \
    --debug_tensorboard=${DEBUG_TENSORBOARD}


