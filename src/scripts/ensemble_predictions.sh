DETECTIONS_FILES="predictions/1_aofinal_frcnn_500_256_extrafeat.tfrecord," \
"predictions/2_aofinal_frcnn_500_256_extrafeat_redsoftmax.tfrecord," \
"predictions/3_aofinal_frcnn_500_256_extrafeat_weights.tfrecord," \
"predictions/4_aofinal_frcnn_500_256_extrafeat_redsoftmax_bikes_higherlr.tfrecord," \
"predictions/9_aofinal_frcnn_500_256_extrafeat_redsoftmax_cars.tfrecord," \
"predictions/10_aofinal_frcnn_500_256_extrafeat_redsoftmax_peds.tfrecord," \
"predictions/11_aofinal_frcnn_500_256_extrafeat_redsoftmax_bikesonly.tfrecord," \
"predictions/12_aofinal_frcnn_500_256_extrafeat_886.tfrecord," \
"predictions/13_aofinal_aofinal_frcnn_500_256_extrafeat_1280.tfrecord," \
"predictions/14_aofinal_frcnn_500_256_extrafeat_redsoftmax_886.tfrecord," \
"predictions/15_aofinal_frcnn_500_256_extrafeat_redsoftmax_1280.tfrecord," \
"predictions/16_aofinal_frcnn_500_256_extrafeat_weights_886.tfrecord," \
"predictions/17_aofinal_frcnn_500_256_extrafeat_weights_1280.tfrecord"
IGNORE_VEHICLES="predictions/10_aofinal_frcnn_500_256_extrafeat_redsoftmax_peds.tfrecord," \
"predictions/11_aofinal_frcnn_500_256_extrafeat_redsoftmax_bikesonly.tfrecord"
IGNORE_PEDESTRIANS="predictions/9_aofinal_frcnn_500_256_extrafeat_redsoftmax_cars.tfrecord," \
"predictions/11_aofinal_frcnn_500_256_extrafeat_redsoftmax_bikesonly.tfrecord"
IGNORE_CYCLISTS="predictions/9_aofinal_frcnn_500_256_extrafeat_redsoftmax_cars.tfrecord," \
"predictions/10_aofinal_frcnn_500_256_extrafeat_redsoftmax_peds.tfrecord"
OUTPUT_PATH=./
IOU_THRESHOLD=0.7
ENSEMBLE_TYPE=NMS

python src/ensemble_predictions.py \
    --detections_files=${DETECTIONS_FILES} \
    --output_path=${OUTPUT_PATH} \
    --iou_threshold=${IOU_THRESHOLD} \
    --ensemble_type=${ENSEMBLE_TYPE} \
    --ignore_vehicles=${IGNORE_VEHICLES} \
    --ignore_pedestrians=${IGNORE_PEDESTRIANS} \
    --ignore_cyclists=${IGNORE_CYCLISTS}