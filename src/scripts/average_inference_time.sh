MODEL=optimized_faster_rcnn
INFERENCE_GRAPH_PATH=saved_models/inference_models/${MODEL}/frozen_inference_graph.pb
METRICS_FILE=predictions/${MODEL}/metrics.csv
DATA_FOLDER=data/camera_data
GPU_DEVICE=0
NUM_ADDITIONAL_CHANNELS=0

python scripts/objdet_api/average_inference_time.py \
      --inference_graph_path=${INFERENCE_GRAPH_PATH} \
      --metrics_file=${METRICS_FILE} \
      --dataset_file_pattern=${DATA_FOLDER}/training/* \
      --num_additional_channels=${NUM_ADDITIONAL_CHANNELS} \
      --gpu_device=${GPU_DEVICE} \



