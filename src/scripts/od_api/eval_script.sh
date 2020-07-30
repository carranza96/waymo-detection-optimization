MODEL_NAME=optimized_faster_rcnn
CHECKPOINT_DIR=saved_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${CHECKPOINT_DIR}/pipeline.config
MODEL_DIR=${CHECKPOINT_DIR}
GPU_DEVICE=0
EVAL_TRAINING_DATA=False
SAMPLE_1_OF_N_EVAL_ON_TRAIN_EXAMPLES=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1

python src/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --eval_training_data=${EVAL_TRAINING_DATA} \
    --sample_1_of_n_eval_on_train_examples=${SAMPLE_1_OF_N_EVAL_ON_TRAIN_EXAMPLES} \
    --sample_1_of_n_eval_examples=${SAMPLE_1_OF_N_EVAL_EXAMPLES} \
    --checkpoint_dir=${CHECKPOINT_DIR} \
    --run_once=True \
    --gpu_device=${GPU_DEVICE}