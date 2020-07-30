# Parameters to be changed
MODEL_NAME=aofinal_frcnn_500_256_extrafeat_1280_3
GPU_DEVICE=0
WRITE_GROUND_TRUTHS=False   # If false, predictions/ground_truths.bin must exist
SPLIT=testing # Dataset split to infer detections
TF_RECORD_FILES=$(ls data/camera_data/${SPLIT}/*.record* | tr '\n' ',') # Files from which to infer detections

# Export inference graph
echo "------------------------------------"
echo "EXPORTING INFERENCE GRAPH"
echo "------------------------------------"

MODEL_DIR=saved_models/final_models/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config
OUTPUT_DIR=saved_models/final_models/inference_models/${MODEL_NAME}

python scripts/objdet_api/export_inference_graph.py \
    --input_type=image_tensor \
    --model_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --output_directory=${OUTPUT_DIR} \
    --gpu_device=${GPU_DEVICE} \




# Infer detections in TFRecord Format
echo "------------------------------------"
echo "INFER DETECTIONS TO TFRECORD"
echo "------------------------------------"

OUTPUT_TFRECORD_PATH=predictions_test/${MODEL_NAME}/${SPLIT}_detections.tfrecord
INFERENCE_GRAPH=saved_models/final_models/inference_models/${MODEL_NAME}/frozen_inference_graph.pb

python scripts/objdet_api/infer_detections.py \
  --input_tfrecord_paths=${TF_RECORD_FILES} \
  --output_tfrecord_path=${OUTPUT_TFRECORD_PATH} \
  --inference_graph=${INFERENCE_GRAPH} \
  --discard_image_pixels \
  --gpu_device=${GPU_DEVICE}



# Convert TFRecord to serialized .bin in metrics.Objects format
echo "------------------------------------"
echo "WRITING SERIALIZED PREDICTIONS"
echo "------------------------------------"

DETECTIONS_FILE=${OUTPUT_TFRECORD_PATH}

python scripts/waymo_eval/write_predictions_file.py \
    --detections_file=${DETECTIONS_FILE} \
    --write_ground_truths=${WRITE_GROUND_TRUTHS}


#################################################################
#################################################################

