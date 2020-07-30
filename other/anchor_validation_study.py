# Import packages
import numpy as np
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from object_detection.core import box_list_ops
from object_detection.core.box_list import BoxList
from utils_tf_record.read_dataset_utils import parse_camera_tfrecord_example
from object_detection.utils.visualization_utils import draw_bounding_boxes_on_image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.enable_eager_execution()

MODEL_DIR = 'saved_models/finetune_best_model/inference_models/ao_frcnn_500_256/'
DATA_FILENAME = 'data/sample_rgb/validation_eval.record'
OUTPUT_FILENAME = 'validation_study.csv'
GENERATE_DATASET = True
NUMBER_IMAGES_TO_SAVE = 100
SAVE_IMAGES_DIR = "wrong_predictions_ao_frcnn_500_256/"


def generate_dataset():
    # Load the Tensorflow model into memory.
    def load_detection_graph(frozen_graph_path):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name='')
                # tf.train.import_meta_graph(graph_def)
        return detection_graph

    def plot_image_with_boxes(image, boxes):
        """Plot a cmaera image."""
        fig, ax = plt.subplots(1, figsize=(20,15))
        ax.imshow(image)
        ax.grid(False)
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            xy = (xmin * image.shape[1], ymin * image.shape[0])
            w, h = (xmax-xmin) * image.shape[1], (ymax-ymin) * image.shape[0]
            rect = patches.Rectangle(xy, w, h, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        return fig

    description_features = [
        "image/height",
        "image/width",
        "image/source_id",
        "image/context_name",
        "image/frame_timestamp_micros",
        "image/time_of_day",
        "image/location",
        "image/weather"
    ]


    def get_features(tf_record):
        description = {}
        for feature in description_features:
            if isinstance(tf_record[feature].numpy(), bytes):
                description[feature] = tf_record[feature].numpy().decode()
            elif isinstance(tf_record[feature].numpy(), np.int64) and feature not in ["image/frame_timestamp_micros"]:
                description[feature] = tf_record[feature].numpy().astype(np.int16)
            else:
                description[feature] = tf_record[feature].numpy()
        description['num_objects'] = len(tf_record["image/object/class/label"])
        return description


    def export_boxes_csv(sess, detection_graph, data_filename, output_filename):
        columns = description_features + ['num_objects'] + ['num_detections', 'xmin', 'ymin', 'xmax', 'ymax', 'iou', 'detected', 'label', 'difficulty']
        def _restart_res():
            return pd.DataFrame(columns=columns)
        res = _restart_res()
        res.to_csv(output_filename)
        dataset = tf.data.TFRecordDataset(data_filename, compression_type='')

        num_images_saved = 0

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        for i, data in enumerate(tqdm(dataset)):

            tf_record = parse_camera_tfrecord_example(data)
            features = get_features(tf_record)

            image = tf.image.decode_jpeg(tf_record["image/encoded"]).numpy()
            image_expanded = np.expand_dims(np.array(image), axis=0)
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            (boxes, n_detections) = sess.run(
                [detection_boxes, num_detections],
                feed_dict={image_tensor: image_expanded})

            boxes = np.squeeze(boxes)[:int(n_detections)]
            gt_boxes = tf.concat([
                tf.expand_dims(tf_record["image/object/bbox/ymin"], axis=1),
                tf.expand_dims(tf_record["image/object/bbox/xmin"], axis=1),
                tf.expand_dims(tf_record["image/object/bbox/ymax"], axis=1),
                tf.expand_dims(tf_record["image/object/bbox/xmax"], axis=1)
            ], axis=1)
            gt_boxes_list = BoxList(gt_boxes)
            boxes_list = BoxList(tf.constant(boxes))
            iou_values = box_list_ops.iou(gt_boxes_list, boxes_list)
            iou_values = tf.math.reduce_max(iou_values, axis=1).numpy()

            features['num_detections'] = int(n_detections)

            labels = tf_record["image/object/class/label"].numpy()
            difficulty_ls = tf_record["image/object/difficult"].numpy()

            missed_boxes = []
            for box, iou, label, difficulty, xmin, ymin, xmax, ymax in zip(gt_boxes.numpy(),
                                                                           iou_values,
                                                                           labels,
                                                                           difficulty_ls,
                                                                           tf_record["image/object/bbox/xmin"],
                                                                           tf_record["image/object/bbox/ymin"],
                                                                           tf_record["image/object/bbox/xmax"],
                                                                           tf_record["image/object/bbox/ymax"]
                                                                           ):
                _row = features.copy()
                _row['xmin'] = xmin.numpy().astype(np.float16)
                _row['ymin'] = ymin.numpy().astype(np.float16)
                _row['xmax'] = xmax.numpy().astype(np.float16)
                _row['ymax'] = ymax.numpy().astype(np.float16)
                _row['iou'] = iou.astype(np.float16)
                iou_threshold = {1: 0.7, 2: 0.5, 3: 0.5}
                _row['detected'] = float(iou) >= iou_threshold[int(label)]
                _row['label'] = label.astype(np.int8)
                _row['difficulty'] = difficulty.astype(np.int8)
                res = res.append(_row, ignore_index=True)

                if num_images_saved < NUMBER_IMAGES_TO_SAVE and float(iou) < iou_threshold[int(label)]:
                    missed_boxes.append([float(xmin.numpy()), float(ymin.numpy()), float(xmax.numpy()), float(ymax.numpy())])

            if i%5 and num_images_saved < NUMBER_IMAGES_TO_SAVE and missed_boxes:
                num_images_saved += 1
                fig = plot_image_with_boxes(image, missed_boxes)
                fig_name = tf_record["image/source_id"].numpy().decode() + '.png'
                fig.tight_layout()
                fig.savefig(os.path.join(SAVE_IMAGES_DIR, fig_name))

            res.to_csv(output_filename, mode='a', header=False)
            res = _restart_res()

        return res

    detection_graph = load_detection_graph(MODEL_DIR + 'frozen_inference_graph.pb')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=detection_graph, config=config)

    _ = export_boxes_csv(sess, detection_graph, DATA_FILENAME, OUTPUT_FILENAME)


if GENERATE_DATASET:
    generate_dataset()

#df = pd.read_csv(OUTPUT_FILENAME, index_col=0)