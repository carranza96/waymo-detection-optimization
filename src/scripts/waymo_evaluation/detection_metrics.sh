MODEL_NAME=optimized_faster_rcnn
PREDICTIONS_FILE=predictions/${MODEL_NAME}/validation_predictions.bin
GTS_FILE=predictions/validation_ground_truths.bin
METRICS_FILE=predictions/${MODEL_NAME}/metrics.csv

echo "------------------------------------"
echo "CALCULATING 2D DETECTION METRICS"
echo "------------------------------------"

cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/compute_detection_metrics_main
cd ..

waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/compute_detection_metrics_main \
${PREDICTIONS_FILE} ${GTS_FILE} > ${METRICS_FILE}

python src/utils/parse_waymo_metrics.py --metrics_file=${METRICS_FILE}

