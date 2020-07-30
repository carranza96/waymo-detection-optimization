MODEL_NAME=aofinal_frcnn_500_256_extrafeat_focalsoftmax
SPLIT=testing # Validation or test
DETECTIONS_FILE=predictions_test/${MODEL_NAME}/${SPLIT}_detections.tfrecord
# OUTPUT_PATH="" # Default to same directory
WRITE_GROUND_TRUTHS=False # If false, predictions/ground_truths.bin must exist

python scripts/waymo_eval/write_predictions_file.py \
    --detections_file=${DETECTIONS_FILE} \
    --output_path=${OUTPUT_PATH} \
    --write_ground_truths=${WRITE_GROUND_TRUTHS} \

