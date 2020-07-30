MODEL=aofinal_frcnn_500_256_extrafeat_focalsoftmax
SPLIT=testing


INPUT_FILENAME=predictions_test/${MODEL}/${SPLIT}_predictions.bin
OUTPUT_FILENAME=predictions_test/${MODEL}/${SPLIT}_submission.bin
SUBMISSION_METADATA=predictions_test/${MODEL}/submission.txtpb  # This file must be edited before executing the script

cd waymo-open-dataset
bazel build waymo_open_dataset/metrics/tools/create_submission
cd ..


waymo-open-dataset/bazel-bin/waymo_open_dataset/metrics/tools/create_submission  \
--input_filenames=${INPUT_FILENAME} \
--output_filename=${OUTPUT_FILENAME} \
--submission_filename=${SUBMISSION_METADATA} \
--num_shards=1

#tar cvf /tmp/my_model.tar /tmp/my_model
#gzip /tmp/my_model.tar
