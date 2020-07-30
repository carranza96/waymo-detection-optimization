MODEL_NAME=optimized_faster_rcnn
MODEL_DIR=saved_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config
OUTPUT_DIR=saved_models/inference_models/${MODEL_NAME}
GPU_DEVICE=0
NUM_ADDITIONAL_CHANNELS=0
#CHECKPOINT=1229837 # Latest checkpoint in model_dir will be selected by default


python src/export_inference_graph.py \
    --input_type=image_tensor \
    --model_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --output_directory=${OUTPUT_DIR} \
    --gpu_device=${GPU_DEVICE} \
    --input_shape=-1,-1,-1,$((3+NUM_ADDITIONAL_CHANNELS))
    #--trained_checkpoint_prefix=${MODEL_DIR}/model.ckpt-${CHECKPOINT}\


