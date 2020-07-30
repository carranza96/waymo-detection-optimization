import os
import csv
from object_detection.eval_util import visualize_detection_results, get_evaluators, get_eval_metric_ops_for_evaluators, \
    evaluator_options_from_eval_config
# from object_detection.utils.visualization_utils import
from src.utils import tf_example_parser
from object_detection.core import standard_fields
from object_detection.utils import label_map_util
from object_detection.protos import eval_pb2
import tensorflow as tf
from object_detection.core import standard_fields as fields
from object_detection.core import box_list
from object_detection.core import box_list_ops
from object_detection.utils import shape_utils
from tqdm import tqdm

tf.enable_eager_execution()
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def write_metrics(metrics, output_dir):
    """Write metrics to the output directory.
    Args:
      metrics: A dictionary containing metric names and values.
      output_dir: Directory to write metrics to.
    """
    tf.logging.info('Writing metrics.')

    with open(os.path.join(output_dir, 'metrics.csv'), 'w') as csvfile:
        metrics_writer = csv.writer(csvfile, delimiter=',')
        for metric_name, metric_value in metrics.items():
            metrics_writer.writerow([metric_name, str(metric_value)])


# FILENAME = "camera_data/training/training.record-00000-of-00075"
FILENAME = "../../old/predictions_v0/initial_crop_size_28/validation_detections.tfrecord-00000-of-00001"
data_parser = tf_example_parser.TfExampleDetectionAndGTParser()
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

# serialized_example = next(iter(dataset))
categories = label_map_util.create_categories_from_labelmap("../../data/camera_data/label_map.pbtxt")

eval_config = eval_pb2.EvalConfig()
eval_config.metrics_set.extend(['coco_detection_metrics'])
# Per category metrics not working
# eval_config.include_metrics_per_category = True

evaluator_options = evaluator_options_from_eval_config(eval_config)
object_detection_evaluators = get_evaluators(eval_config, categories, evaluator_options)
object_detection_evaluator = object_detection_evaluators[0]


def scale_boxes_to_absolute_coordinates(decoded_dict):
    def _scale_box_to_absolute(args):
        boxes, height, width = args
        return box_list_ops.to_absolute_coordinates(
            box_list.BoxList(boxes), height, width).get()

    decoded_dict[fields.InputDataFields.groundtruth_boxes] = _scale_box_to_absolute(
        (tf.convert_to_tensor(decoded_dict[fields.InputDataFields.groundtruth_boxes], dtype=tf.float32),
         decoded_dict[fields.TfExampleFields.height], decoded_dict[fields.TfExampleFields.width])).numpy()

    decoded_dict[fields.DetectionResultFields.detection_boxes] = _scale_box_to_absolute(
        (tf.convert_to_tensor(decoded_dict[fields.DetectionResultFields.detection_boxes], dtype=tf.float32),
         decoded_dict[fields.TfExampleFields.height], decoded_dict[fields.TfExampleFields.width])).numpy()

    return decoded_dict


for i, serialized_example in tqdm(enumerate(dataset)):
    example = tf.train.Example.FromString(serialized_example.numpy())
    decoded_dict = data_parser.parse(example)

    decoded_dict = scale_boxes_to_absolute_coordinates(decoded_dict)

    object_detection_evaluator.add_single_ground_truth_image_info(
        decoded_dict[standard_fields.DetectionResultFields.key], decoded_dict)
    object_detection_evaluator.add_single_detected_image_info(
        decoded_dict[standard_fields.DetectionResultFields.key], decoded_dict)

    #
    # if i == 10:
    #     break

metrics = object_detection_evaluator.evaluate()
# write_metrics(metrics, "evaluation/")
# visualize_detection_results()

# mAP = metrics['DetectionBoxes_Precision/mAP']
# print(mAP)


#  result_dict_for_single_example(image, key, detections, groundtruth=None, class_agnostic=False, scale_to_absolute=False
# def visualize_detection_results(result_dict, tag,global_step,categories, summary_dir='',
# export_dir='', agnostic_mode=False, show_groundtruth=False, groundtruth_box_visualization_color='black',
# min_score_thresh=.5, max_num_predictions=20,skip_scores=False, skip_labels=False, keep_image_id_for_visualization_export=False):
