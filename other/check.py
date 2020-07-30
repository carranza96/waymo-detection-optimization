import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils_tf_record.read_dataset_utils import read_frame_waymo_segment,\
    read_and_parse_sharded_dataset, parse_camera_tfrecord_example, get_dataset_class_distribution
from waymo_open_dataset.protos import metrics_pb2,submission_pb2
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()


FILENAME = "data/sample_rgb/training/training.record-00000-of-00798"
FILENAME = "data/sample_rgb/training_eval.record"
FILENAME = "data/camera_data/testing/testing.record-00000-of-00150"
FILENAME = "predictions_test/23_aofinal_frcnn_500_256_extrafeat_weights_1280_tta1.2.tfrecord"
# FILENAME = "data/camera_data_testing/testing/testing.record-00020-of-00150"
#FILENAME = "predictions_test/high_res_modified_lr(web)/testing_detections.tfrecord"
FILENAME = "predictions_test/rgbR/testing_detections.tfrecord"
FILENAME = "data/camera_frontal3/training/frontal_training.record-00000-of-00798"

# # FILENAME = "camera_data/training/training.record-00000-of-00798"
# more_channels = False
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

# FILENAME_PATTERN = "data/camera_data_rgb_rie/training/*"
# ignore_order = tf.data.Options()
# ignore_order.experimental_deterministic = False
# dataset = tf.data.Dataset.list_files(FILENAME_PATTERN)
# dataset = dataset.with_options(ignore_order)
# dataset = dataset.interleave(tf.data.TFRecordDataset,
#                                  cycle_length=32,
#                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)


# example = next(iter(dataset))
# parsed = tf.train.Example.FromString(example.numpy())
# feature = parsed.features.feature

# classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
# classes = feature['image/object/class/label'].int64_list.value
# detections = feature['image/detection/label'].int64_list.value
# detection_scores = feature['image/detection/score'].float_list.value
#
c = 0
cars,peds, bikes = 0,0,0
contexts = []
source_ids = []
for i, raw_example in enumerate(dataset):
    # if i>=100:
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature
        if i%1000==0:
            print(i)

        classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
        classes = feature['image/detection/label'].int64_list.value

        source_id = feature['image/source_id'].bytes_list.value[0].decode()
        context_name = feature['image/context_name'].bytes_list.value[0].decode()
        frame = feature['image/frame_timestamp_micros'].int64_list.value[0]
        height = feature['image/height'].int64_list.value[0]
        width = feature['image/width'].int64_list.value[0]
        channels = feature['image/channels'].int64_list.value[0]
        #raw_img = feature['image/encoded'].bytes_list.value[0]
        # raw_additional_channels = feature['image/additional_channels/encoded'].bytes_list.value[0]
        # if context_name=="17958696356648515477_1660_000_1680_000":
        #     print(frame)
        #image = tf.image.decode_jpeg(raw_img)

            # if more_channels:
            #     # additional_channels = tf.reshape(tf.io.decode_raw(raw_img, out_type=tf.uint8), shape=[height, width, channels])
            #     additional_channels = tf.image.decode_jpeg(raw_additional_channels)
            #
            # if 'lidar' in FILENAME:
            #     lidar_channel = image.numpy().copy()[:, :, 0].reshape((image.shape[0], image.shape[1]))
            #     plt.figure()
            #     ax = plt.subplot(1, 1, 1)
            #     plt.imshow(lidar_channel, cmap='Reds')
            #     plt.title(feature["image/camera_name"].int64_list.value[0])
            #     plt.grid(False)
            #     plt.axis('off')
            #     plt.show()



        xmins = [x*width for x in feature['image/detection/bbox/xmin'].float_list.value]
        xmaxs = [x*width for x in feature['image/detection/bbox/xmax'].float_list.value]
        ymins = [x*height for x in feature['image/detection/bbox/ymin'].float_list.value]
        ymaxs = [x*height for x in feature['image/detection/bbox/ymax'].float_list.value]

            #
            # labels_h = [ymax-ymin for ymax, ymin in zip(ymaxs, ymins)]
            # labels_w = [xmax-xmin for xmax, xmin in zip(xmaxs, xmins)]
            #
            # if any([h < 2 or w < 2 for h, w in zip(labels_h, labels_w)]):
            #
            #     fig = plt.figure()
            #     ax = plt.subplot(1, 1, 1)
            #     plt.imshow(image)
            #     for z in range(len(xmins)):
            #         if labels_h[z] < 2 or labels_w[z] < 2:
            #             ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                            width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                            linewidth=1, edgecolor='red', facecolor='none'))
            #
            #
            #     plt.title(feature["image/camera_name"].int64_list.value[0])
            #     plt.grid(False)
            #     plt.axis('off')
            #     fig.savefig('fig1.svg', format='svg', dpi=1200)
            #     plt.show()
        #
        # fig2 = plt.figure()
        # ax = plt.subplot(1, 1, 1)
        # plt.imshow(image)
        #
        # for z in range(len(xmins)):
        #     ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
        #                                        width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
        #                                        linewidth=1, edgecolor='red', facecolor='none'))
        #
        # plt.title(feature["image/camera_name"].int64_list.value[0])
        # plt.grid(False)
        # plt.axis('off')
        # # fig2.savefig('fig4.svg', format='svg', dpi=1200)
        # plt.show()


                # if more_channels:
                #     for i in range(additional_channels.shape[2]):
                #         plt.figure()
                #         ax = plt.subplot(1, 1, 1)
                #         plt.imshow(tf.reshape(additional_channels[..., i], (additional_channels.shape[0],additional_channels.shape[1])),
                #                    cmap='Reds_r')
                #         plt.show()



                    # plt.figure()
                    # ax = plt.subplot(1, 1, 1)
                    # plt.imshow(tf.concat([image, additional_channels], axis=2))
                    # plt.show()

        c += len(xmins)
        bikes += sum([1 for x in classes if x==3])
        cars += sum([1 for x in classes if x == 1])
        peds += sum([1 for x in classes if x == 2])

        contexts.append(context_name)
        if len(xmins) > 0:
            source_ids.append(source_id)
        #break

print(c)
print(cars, peds, bikes)
print(np.unique(source_ids).shape)

#
# f = open('predictions_test/aofinal_frcnn_500_256_extrafeat_weights/testing_predictions.bin','rb')
# predictions = metrics_pb2.Objects()
# predictions.ParseFromString(f.read())
# print("Num boxes:", len(predictions.objects))
# print("Num contexts", np.unique([p.context_name for p in predictions.objects]).shape)
# a = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in predictions.objects]
# print("Num images:", np.unique(a, axis=0).shape)

# f = open('predictions_test/high_res_modified_lr(web)/testing_predictions.bin','rb')
# predictions2 = metrics_pb2.Objects()
# predictions2.ParseFromString(f.read())
# print("Num boxes:", len(predictions2.objects))
# print("Num contexts", np.unique([p.context_name for p in predictions2.objects]).shape)
# d = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in predictions2.objects]
# print("Num images:", np.unique(d, axis=0).shape)

# np.unique(np.concatenate((a,b)),axis=0).shape
#
# f = open('predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_bikes_higherlr/testing_predictions.bin','rb')
# predictions2 = metrics_pb2.Objects()
# predictions2.ParseFromString(f.read())
# print("Num boxes:", len(predictions2.objects))
# print("Num contexts", np.unique([p.context_name for p in predictions2.objects]).shape)
# b = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in predictions2.objects]
# print("Num images:", np.unique(b, axis=0).shape)

# f = open('predictions_test/cascade_predictions/testing_predictions.bin', 'rb')
# predictions2 = metrics_pb2.Objects()
# predictions2.ParseFromString(f.read())
# print("Num boxes:", len(predictions2.objects))
# print("Num contexts", np.unique([p.context_name for p in predictions2.objects]).shape)
# d = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in predictions2.objects]
# print("Num images:", np.unique(d, axis=0).shape)

# f = open('predictions_test/ensemble_frcnn_23models/testing_submission.bin', 'rb')
# submission = submission_pb2.Submission()
# submission.ParseFromString(f.read())
# print("Num boxes:", len(submission.inference_results.objects))
# print("Num contexts", np.unique([p.context_name for p in submission.inference_results.objects]).shape)
# c = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in submission.inference_results.objects]
# print("Num images:", np.unique(c, axis=0).shape)
# types = Counter([x.object.type for x in submission.inference_results.objects])
# print(types)

# f = open('predictions_test/cascade_predictions/cascade_submission.bin', 'rb')
# submission2 = submission_pb2.Submission()
# submission2.ParseFromString(f.read())
# print("Num boxes:", len(submission2.inference_results.objects))
# print("Num contexts", np.unique([p.context_name for p in submission2.inference_results.objects]).shape)
# c = [(p.context_name, p.frame_timestamp_micros, p.camera_name) for p in submission2.inference_results.objects]
# print("Num images:", np.unique(c, axis=0).shape)
# types = Counter([x.object.type for x in submission2.inference_results.objects])
# print(types)




















