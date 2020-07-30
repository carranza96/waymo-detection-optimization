MODEL_NAME=optimized_faster_rcnn
SPLIT=validation
#TF_RECORD_FILES=$(ls camera_data/${SPLIT}/${SPLIT}.record-0000[0-1]-of-00025 | tr '\n' ',')
TF_RECORD_FILES=$(ls data/camera_data/${SPLIT}/${SPLIT}.record* | tr '\n' ',')
OUTPUT_TFRECORD_PATH=predictions/${MODEL_NAME}/${SPLIT}_detections.tfrecord
INFERENCE_GRAPH=saved_models/inference_models/${MODEL_NAME}/frozen_inference_graph.pb
GPU_DEVICE=0
NUM_ADDITIONAL_CHANNELS=0


python src/infer_detections.py \
  --input_tfrecord_paths=${TF_RECORD_FILES} \
  --output_tfrecord_path=${OUTPUT_TFRECORD_PATH} \
  --inference_graph=${INFERENCE_GRAPH} \
  --discard_image_pixels \
  --gpu_device=${GPU_DEVICE} \
  --num_additional_channels=${NUM_ADDITIONAL_CHANNELS}

