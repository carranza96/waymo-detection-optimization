# Parameters to be changed
declare -a StringArray=("faster_rcnn_resnet101_v1_1280x1920_coco17_tpu-8")
for MODEL_NAME in ${StringArray[@]}; do
#for MODEL_NAME in $(ls saved_models/jesus/); do
#echo $MODEL_NAME

GPU_DEVICE=0
WRITE_GROUND_TRUTHS=True   # If false, predictions/validation_ground_truths.bin must exist
SPLIT=validation  # Dataset split to infer detections
DATA_FOLDER=/home/guest/Escritorio/TFM/camera_data_v0
NUM_ADDITIONAL_CHANNELS=0

echo "------------------------------------"
echo "EXPORTING INFERENCE GRAPH"
echo "------------------------------------"

MODEL_DIR=saved_models/jesus/${MODEL_NAME}
PIPELINE_CONFIG_PATH=${MODEL_DIR}/pipeline.config
OUTPUT_DIR=saved_models/inference_models/${MODEL_NAME}

python tf2/exporter_main_v2.py \
    --input_type=image_tensor \
    --trained_checkpoint_dir=${MODEL_DIR} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --output_directory=${OUTPUT_DIR} \
    --gpu_device=${GPU_DEVICE} \




# Infer detections in TFRecord Format
echo "------------------------------------"
echo "INFER DETECTIONS TO TFRECORD"
echo "------------------------------------"

OUTPUT_TFRECORD_PATH=predictions/${MODEL_NAME}/${SPLIT}_detections.tfrecord
INFERENCE_GRAPH=saved_models/inference_models/${MODEL_NAME}/saved_model/
TF_RECORD_FILES=$(ls ${DATA_FOLDER}/${SPLIT}/${SPLIT}.record* | tr '\n' ',') # Files from which to infer detections
#TF_RECORD_FILES=${DATA_FOLDER}/${SPLIT}_eval.record
#
python tf2/infer_detections_v2.py \
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

python src/write_predictions_file.py \
    --detections_file=${DETECTIONS_FILE} \
    --write_ground_truths=${WRITE_GROUND_TRUTHS} \


# Compute metrics
echo "------------------------------------"
echo "CALCULATING 2D DETECTION METRICS"
echo "------------------------------------"

PREDICTIONS_FILE=predictions/${MODEL_NAME}/validation_predictions.bin
GTS_FILE=predictions//${MODEL_NAME}/validation_ground_truths.bin
METRICS_FILE=predictions/${MODEL_NAME}/metrics.csv

cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cd ..

waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}

python src/utils/parse_waymo_metrics.py --metrics_file=${METRICS_FILE}



# Inference time
echo "------------------------------------"
echo "CALCULATING AVERAGE INFERENCE TIME"
echo "------------------------------------"


python tf2/average_inference_time_v2.py \
      --input_tfrecord_paths=${TF_RECORD_FILES} \
      --inference_graph_path=${INFERENCE_GRAPH} \
      --metrics_file=${METRICS_FILE} \
      --gpu_device=${GPU_DEVICE} \
      --num_additional_channels=${NUM_ADDITIONAL_CHANNELS} \

done