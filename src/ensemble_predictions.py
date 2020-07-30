import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import patches
from utils_tf_record.read_dataset_utils import parse_camera_tfrecord_example, write_serialized_string
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset import label_pb2
from absl import app
from absl import flags
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.enable_eager_execution()

FLAGS = flags.FLAGS
flags.DEFINE_list('detections_files',
                    ["predictions_final_sample/aofinal_frcnn_500_256_extrafeat_red_softmax_bikes_higherlr/validation_detections.tfrecord",
                    "predictions_final_sample/aofinal_frcnn_500_256_extrafeat/validation_detections.tfrecord",
                     "predictions_final_sample/aofinal_frcnn_500_256_extrafeat_redsoftmax/validation_detections.tfrecord",
                     "predictions_final_sample/aofinal_frcnn_500_256_extrafeat_weights/validation_detections.tfrecord"
                     ],
                    'TFRecord detections files separated by comma')
flags.DEFINE_list('ignore_vehicles', [], 'detection_files of models which should ignore vehicle predictions.')
flags.DEFINE_list('ignore_pedestrians', [], 'detection_files of models which should ignore pedestrian predictions.')
flags.DEFINE_list('ignore_cyclists', [], 'detection_files of models which should ignore cyclist predictions.')
flags.DEFINE_string('output_path', "predictions_final_sample/ensemble4",
                    'Path to output serialized .bin files with metrics.Objects proto format')
flags.DEFINE_float("iou_threshold", 0.7, "IOU Threshold used to consider two predictions the same predictions")
flags.DEFINE_string("ensemble_type", 'NMS', 'Ensemble method to use: NMS (non-maximum supression) or AVG (Average the position between same predictions)')


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

def to_numpy(frame, features,  predictions, model=None, ignore_vehicles=False, ignore_pedestrians=False, ignore_cyclists=False):
    image_id = frame['image/context_name'].numpy().decode() + '_' + str(int(frame['image/frame_timestamp_micros'])) + '_' + str(int(frame['image/camera_name']))
    frame_features = {
        "image/context_name": frame['image/context_name'].numpy().decode(),
        "image/frame_timestamp_micros": int(frame['image/frame_timestamp_micros']),
        "image/camera_name": int(frame['image/camera_name']),
        "image/width": int(frame['image/width']),
        "image/height": int(frame['image/height']),
        'predictions': [],
        'predictions_1': [],
        'predictions_2': [],
        'predictions_3': [],
    }

    if image_id not in features:
        features[image_id] = frame_features

    # Predictions
    zip_pd_boxes = zip(frame['image/detection/bbox/xmax'], frame['image/detection/bbox/xmin'],
                       frame['image/detection/bbox/ymax'], frame['image/detection/bbox/ymin'],
                       frame['image/detection/label'], frame["image/detection/score"])

    npreds = len(predictions)

    i = npreds
    for (x_max, x_min, y_max, y_min, label, score) in zip_pd_boxes:
        if int(label.numpy()) == 1 and ignore_vehicles:
            continue
        if int(label.numpy()) == 2 and ignore_pedestrians:
            continue
        if int(label.numpy()) == 3 and ignore_cyclists:
            continue
        row = [
            y_min.numpy(),
            x_min.numpy(),
            y_max.numpy(),
            x_max.numpy(),
            label.numpy(),
            score.numpy()
        ]
        predictions.append(row)
        features[image_id]['predictions'] += [i]
        features[image_id]['predictions_{}'.format(int(label.numpy()))] += [i]
        i += 1

    return features, predictions


def AvgEnsemble(dets, iou_threshold=0.5):
    """
    Args:
        dets: list of [ymin, xmin, ymax, xmax, class, score]
        iou_threshold: float

    Returns: new detections like a list of [ymin, xmin, ymax, xmax, class, score]
    """

    def computeIOU(box1, box2):
        """
        Args:
            box1:  [ymin, xmin, ymax, xmax]
            box2:  [ymin, xmin, ymax, xmax]
        Returns: iou between box1 and box2
        """
        y11, x11, y12, x12 = box1
        y21, x21, y22, x22 = box2

        x_left = max(x11, x21)
        y_top = max(y11, y21)
        x_right = min(x12, x22)
        y_bottom = min(y12, y22)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersect_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x12 - x11) * (y12 - y11)
        box2_area = (x22 - x21) * (y22 - y21)

        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou

    keep = []
    used = []
    for box1 in dets:
        if box1 in used:
            continue
        used.append(box1)

        found = [box1]
        for box2 in dets:
            if box2[4] != box1[4] or box2 in used :
                # Skip if box already used or box of different class
                continue
            iou = computeIOU(box1[:4], box2[:4])
            if iou > iou_threshold:
                used.append(box2)
                found.append(box2)

        y1_sum = 0.
        x1_sum = 0.
        y2_sum = 0.
        x2_sum = 0.
        weight_sum = 0.
        score = 1.

        for b in found:
            y1_sum += b[0] * b[5]
            x1_sum += b[1] * b[5]
            y2_sum += b[2] * b[5]
            x2_sum += b[3] * b[5]
            weight_sum += b[5]
            score *= (1 - b[5])

        y1 = y1_sum/weight_sum
        x1 = x1_sum/weight_sum
        y2 = y2_sum/weight_sum
        x2 = x2_sum/weight_sum

        score = 1-score

        label = int(box1[4])

        new_box = [y1, x1, y2, x2, label, score]
        keep.append(new_box)
    return keep


def main(_):
    detections_files = FLAGS.detections_files
    IOU_THRESHOLD = FLAGS.iou_threshold
    ENSEMBLE_TYPE = FLAGS.ensemble_type
    IGNORE_VEHICLES = FLAGS.ignore_vehicles
    IGNORE_CYCLISTS = FLAGS.ignore_cyclists
    IGNORE_PEDESTRIANS = FLAGS.ignore_pedestrians
    assert all(ignore_file in detections_files for ignore_file in IGNORE_VEHICLES + IGNORE_PEDESTRIANS + IGNORE_CYCLISTS), 'error while defining ignore class predictions'
    print("Num models: {}\nNum models ignoring vehicles: {}\nNum models ignoring pedestrians: {}\nNum models ignoring cyclists: {}\nIOU Threshold: {}\nEnsemble type: {}\n".format(len(detections_files), len(IGNORE_VEHICLES), len(IGNORE_PEDESTRIANS), len(IGNORE_CYCLISTS), IOU_THRESHOLD, ENSEMBLE_TYPE))

    # Reading predictions from models
    t0 = time.time()
    detections_datasets = [(detections_file, tf.data.TFRecordDataset(detections_file.strip(), compression_type='') \
                               .map(lambda x: parse_camera_tfrecord_example(x, with_detections=True))) \
                           for detections_file in detections_files]


    detections = []
    features = {}

    for j, (detection_file, detections_dataset) in enumerate(detections_datasets):
        ignore_vehicles = detection_file in IGNORE_VEHICLES
        ignore_pedestrians = detection_file in IGNORE_PEDESTRIANS
        ignore_cyclists = detection_file in IGNORE_CYCLISTS
        for i, frame in tqdm(enumerate(detections_dataset), desc="Reading predictions from model #{}".format(j)):
            features, detections = to_numpy(frame, features, detections, model=str(j), ignore_vehicles=ignore_vehicles,
                                            ignore_pedestrians=ignore_pedestrians, ignore_cyclists=ignore_cyclists)

    detections = np.array(detections)

    num_original_detections = len(detections)

    image_ids = []
    # Merging predictions for each images
    if ENSEMBLE_TYPE == 'NMS':
        keep = []
        for image_id in tqdm(features, desc="Merging predictions for each images (NMS method)"):
            image_features = features[image_id]

            labels = [1, 2, 3]
            for label in labels:
                indices = image_features['predictions_{}'.format(label)]
                image_detection_label = detections[image_features['predictions_{}'.format(label)]]

                boxes = tf.convert_to_tensor(image_detection_label[:, :4], tf.float32)
                scores = tf.convert_to_tensor(image_detection_label[:, 5], tf.float32)
                max_output_size = tf.convert_to_tensor(len(image_detection_label), tf.int32)
                iou_threshold = IOU_THRESHOLD

                filtered_image_detections = tf.image.non_max_suppression(boxes, scores, max_output_size, iou_threshold)
                filtered_image_detections = [indices[x] for x in filtered_image_detections.numpy()]
                keep += filtered_image_detections
                image_ids += [image_id for _ in range(len(filtered_image_detections))]

        detections = detections[keep]

    elif ENSEMBLE_TYPE == 'AVG':
        new_detections = []
        for image_id in tqdm(features, desc="Merging predictions for each images (NMS method)"):
            image_features = features[image_id]
            image_detections = detections[image_features['predictions']]

            context_name, frame_timestamp_micros, camera_name, width, height = image_id[0], image_id[1], image_id[2], image_id[3], image_id[4]

            boxes = [np_box.tolist() for np_box in image_detections]
            new_dets = AvgEnsemble(boxes, IOU_THRESHOLD)

            new_detections += new_dets
            image_ids += [image_id for _ in range(len(new_dets))]

        detections = np.array(new_detections)

    num_detections = len(detections)

    # Creating prediction objects
    predictions = metrics_pb2.Objects()
    for index in tqdm(list(range(len(detections))), desc='Creating prediction objects'):
        image_features = features[image_ids[index]]
        y_min, x_min, y_max,  x_max, label, score = detections[index]
        frame_features = {
            "image/context_name": image_features["image/context_name"],
            "image/frame_timestamp_micros": image_features["image/frame_timestamp_micros"],
            "image/camera_name": image_features["image/camera_name"],
            "image/width": image_features["image/width"],
            "image/height": image_features["image/height"]
        }
        obj = create_object(frame_features, x_max, x_min, y_max, y_min, int(label), difficulty=None,
                            score=score)
        predictions.objects.extend([obj])

    # Write to file
    output_path = FLAGS.output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output_path = os.path.join(output_path, 'ensemble_{}_predictions.bin'.format(ENSEMBLE_TYPE.lower()))
    write_serialized_string(output_path, predictions)

    t1 = time.time()
    print("Detections remaining: {}/{} ({:.2f})".format(num_detections, num_original_detections, num_detections/num_original_detections))
    print("Processing time: {:.2f} s".format(t1-t0))

    return 0


if __name__ == '__main__':
    app.run(main)
