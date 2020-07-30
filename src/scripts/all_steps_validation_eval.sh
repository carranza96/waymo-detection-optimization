# Parameters to be changed
MODEL_NAME=aofinal_frcnn_500_256_extrafeat_redsoftmax_cars
GPU_DEVICE=0
WRITE_GROUND_TRUTHS=True   # If false, predictions/validation_ground_truths.bin must exist
SPLIT=validation  # Dataset split to infer detections
DATA_FOLDER=data/sample_rgb
NUM_ADDITIONAL_CHANNELS=0

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
    --input_shape=-1,-1,-1,$((3+NUM_ADDITIONAL_CHANNELS))





# Infer detections in TFRecord Format
echo "------------------------------------"
echo "INFER DETECTIONS TO TFRECORD"
echo "------------------------------------"

OUTPUT_TFRECORD_PATH=predictions_sample/${MODEL_NAME}/${SPLIT}_detections.tfrecord
INFERENCE_GRAPH=saved_models/final_models/inference_models/${MODEL_NAME}/frozen_inference_graph.pb
#TF_RECORD_FILES=$(ls ${DATA_FOLDER}/${SPLIT}/${SPLIT}.record* | tr '\n' ',') # Files from which to infer detections
TF_RECORD_FILES=${DATA_FOLDER}/${SPLIT}_eval.record

python scripts/objdet_api/infer_detections.py \
  --input_tfrecord_paths=${TF_RECORD_FILES} \
  --output_tfrecord_path=${OUTPUT_TFRECORD_PATH} \
  --inference_graph=${INFERENCE_GRAPH} \
  --discard_image_pixels \
  --gpu_device=${GPU_DEVICE} \
  --num_additional_channels=${NUM_ADDITIONAL_CHANNELS}




# Convert TFRecord to serialized .bin in metrics.Objects format
echo "------------------------------------"
echo "WRITING SERIALIZED PREDICTIONS"
echo "------------------------------------"

DETECTIONS_FILE=${OUTPUT_TFRECORD_PATH}

python scripts/waymo_eval/write_predictions_file.py \
    --detections_file=${DETECTIONS_FILE} \
    --write_ground_truths=${WRITE_GROUND_TRUTHS} \


# Compute metrics
echo "------------------------------------"
echo "CALCULATING 2D DETECTION METRICS"
echo "------------------------------------"

PREDICTIONS_FILE=predictions_sample/${MODEL_NAME}/validation_predictions.bin
GTS_FILE=predictions_sample/validation_ground_truths.bin
METRICS_FILE=predictions_sample/${MODEL_NAME}/metrics.csv

cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cd ..

waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}

python scripts/waymo_eval/parse_metrics_result.py --metrics_file=${METRICS_FILE}



# Inference time
echo "------------------------------------"
echo "CALCULATING AVERAGE INFERENCE TIME"
echo "------------------------------------"

DATASET_FILE_PATTERN=${DATA_FOLDER}/training/*

python scripts/objdet_api/average_inference_time.py \
      --inference_graph_path=${INFERENCE_GRAPH} \
      --metrics_file=${METRICS_FILE} \
      --dataset_file_pattern=${DATASET_FILE_PATTERN} \
      --num_additional_channels=${NUM_ADDITIONAL_CHANNELS}


