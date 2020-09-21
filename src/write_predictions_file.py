from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
from src.utils.read_dataset import parse_camera_tfrecord_example, write_serialized_string
from tqdm import tqdm
import tensorflow as tf
import os
from absl import app
from absl import flags

""" Tool to export predictions to serialized .bin with metrics.Objects proto format 
Required to compute Waymo evaluation metrics locally
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_string('detections_file', "predictions/low_res/validation_detections.tfrecord",
                    'TFRecord file containing ground truths and detections')
flags.DEFINE_string('output_path', None,
                    'Path to output serialized .bin files with metrics.Objects proto format')
flags.DEFINE_boolean('write_ground_truths', False, 'If true, writes also ground_truths.bin file')

# TODO: Accelerate process avoid append to list
# TODO: Frame shared features


def create_frame_objects(frame, write_ground_truths):
    ground_truths, predictions = [], []
    frame_features = get_frame_shared_features(frame)

    # Ground truths
    if write_ground_truths:
        zip_gt_boxes = zip(frame['image/object/bbox/xmax'], frame['image/object/bbox/xmin'],
                           frame['image/object/bbox/ymax'], frame['image/object/bbox/ymin'],
                           frame['image/object/class/label'], frame["image/object/difficult"])

        for x_max, x_min, y_max, y_min, label, difficult in zip_gt_boxes:
            obj = create_object(frame_features, x_max, x_min, y_max, y_min, label.numpy(), difficult.numpy(), score=None)
            ground_truths.append(obj)

    # Predictions
    zip_pd_boxes = zip(frame['image/detection/bbox/xmax'], frame['image/detection/bbox/xmin'],
                       frame['image/detection/bbox/ymax'], frame['image/detection/bbox/ymin'],
                       frame['image/detection/label'], frame["image/detection/score"])

    for x_max, x_min, y_max, y_min, label, score in zip_pd_boxes:
        obj = create_object(frame_features, x_max, x_min, y_max, y_min, label.numpy(), difficulty=None,
                            score=score.numpy())
        predictions.append(obj)

    return ground_truths, predictions


def get_frame_shared_features(frame):
    frame_features = {
        "image/context_name": frame['image/context_name'].numpy().decode(),
        "image/frame_timestamp_micros": int(frame['image/frame_timestamp_micros']),
        "image/camera_name": int(frame['image/camera_name']),
        "image/width": int(frame['image/width']),
        "image/height": int(frame['image/height'])
    }
    return frame_features


def create_object(frame_features, x_max, x_min, y_max, y_min, label, difficulty=None, score=None):
    obj = metrics_pb2.Object()

    def create_label(img_width, img_height, x_max, x_min, y_max, y_min, label_type, difficulty=None):
        lab = label_pb2.Label()
        lab.box.center_x = (x_max + x_min) / 2 * img_width
        lab.box.center_y = (y_max + y_min) / 2 * img_height
        lab.box.length = (x_max - x_min) * img_width
        lab.box.width = (y_max - y_min) * img_height

        lab.type = 4 if label_type == 3 else label_type  # Revert cyclist label to 4
        if difficulty is not None:
            lab.detection_difficulty_level = 1 if difficulty == 0 else difficulty
        return lab

    label = create_label(frame_features['image/width'], frame_features['image/height'],
                         x_max, x_min, y_max, y_min, label, difficulty)
    obj.object.MergeFrom(label)
    if score:
        obj.score = score
    obj.context_name = frame_features["image/context_name"]
    obj.frame_timestamp_micros = frame_features["image/frame_timestamp_micros"]
    obj.camera_name = frame_features["image/camera_name"]

    return obj


def main(_):
    # Check flag file_path
    if not FLAGS.detections_file:
        raise ValueError('file_path must be specified')

    write_ground_truths = FLAGS.write_ground_truths
    detections_file = FLAGS.detections_file

    # detections_file = ["predictions_test/aofinal_frcnn_500_256_extrafeat_2cameras/12_aofinal_frcnn_500_256_extrafeat_886.tfrecord",
    #                    "predictions_test/aofinal_frcnn_500_256_extrafeat_2cameras/13_aofinal_frcnn_500_256_extrafeat_1280.tfrecord"]
    # detections_file = ["predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_2cameras/14_aofinal_frcnn_500_256_extrafeat_redsoftmax_886.tfrecord",
    #                    "predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_2cameras/15_aofinal_frcnn_500_256_extrafeat_redsoftmax_1280.tfrecord"]
    #detections_file = ["predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_2cameras_sample3/aofinal_frcnn_500_256_extrafeat_redsoftmax_886.tfrecord",
     #                   "predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_2cameras_sample3/aofinal_frcnn_500_256_extrafeat_redsoftmax_1280.tfrecord"]

    # detections_file = ["predictions_test/aofinal_frcnn_500_256_extrafeat_2cameras_sample3/aofinal_frcnn_500_256_extrafeat_886_3.tfrecord",
    #                     "predictions_test/aofinal_frcnn_500_256_extrafeat_2cameras_sample3/aofinal_frcnn_500_256_extrafeat_1280_3.tfrecord"]

    detections_dataset = tf.data.TFRecordDataset(detections_file, compression_type='') \
        .map(lambda x: parse_camera_tfrecord_example(x, with_detections=True))

    # Choose default output_path if not defined
    if not FLAGS.output_path:
        output_path = "/"
        output_path = output_path.join(detections_file.split("/")[:-1]) + "/"
    else:
        output_path = FLAGS.output_path

    # output_path = "predictions_test/aofinal_frcnn_500_256_extrafeat_2cameras_sample3/"
    #output_path = "predictions_test/aofinal_frcnn_500_256_extrafeat_redsoftmax_2cameras_sample3/"
    #output_path = "predictions_test/aofinal_frcnn_500_256_extrafeat_weights_2cameras/"

    # Create output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Parse data
    predictions = metrics_pb2.Objects()
    ground_truths = metrics_pb2.Objects()

    for i, frame in tqdm(enumerate(detections_dataset)):
        # if frame['image/time_of_day'].numpy()==b'Night':
            frame_gts, frame_pds = create_frame_objects(frame, write_ground_truths)
            predictions.objects.extend(frame_pds)
            if write_ground_truths:
                ground_truths.objects.extend(frame_gts)

    # Write data
    split = "validation" if "validation" in detections_file else "testing"
    write_serialized_string(output_path + split + "_predictions.bin", predictions)
    if write_ground_truths:
        write_serialized_string(output_path + split + "_ground_truths.bin", ground_truths)

    return 0


if __name__ == '__main__':
    app.run(main)
