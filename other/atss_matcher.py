import tensorflow as tf
import os
from object_detection.core import box_list_ops
from object_detection.core.box_list import BoxList
from object_detection.core.target_assigner import create_target_assigner
from object_detection.core.region_similarity_calculator import RegionSimilarityCalculator
tf.enable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# ymin, xmin, ymax, xmax
gt_boxes_array = tf.convert_to_tensor([[0.0, 0.0, 1.0, 1.0],
                                       [0.0, 0.0, 0.5, 0.5]])
anchors_array = tf.convert_to_tensor([[0.0, 0.0, 0.9, 0.9],
                                      [0.0, 0.0, 0.5, 0.5],
                                      [0.0, 0.0, 0.7, 0.7],
                                      [0.1, 0.1, 0.2, 0.2]])
#anchors = [[0.0, 0.0, 0.5, 0.5],[0.0, 0.0, 0.25, 0.25] ]
gt_boxes = BoxList(tf.convert_to_tensor(gt_boxes_array))
anchors = BoxList(tf.convert_to_tensor(anchors_array))


iou_values = box_list_ops.iou(gt_boxes, anchors)
max_iou_values = tf.math.reduce_max(iou_values, axis=1).numpy()

target_assigner = create_target_assigner('FasterRCNN', 'detection', negative_class_weight=1.0, use_matmul_gather=False)

# Each row is a ground truth box, and each column is an anchor (proposal)
match_quality_matrix = target_assigner._similarity_calc.compare(gt_boxes, anchors)

match = target_assigner._matcher.match(match_quality_matrix)

cls_targets, cls_weights, reg_targets, reg_weights, match_results = \
    target_assigner.assign(anchors, gt_boxes, groundtruth_labels=None,
             unmatched_class_label=None,
             groundtruth_weights=None)


class CenterDistanceSimilarity(RegionSimilarityCalculator):
  """Class to compute similarity based on center L2 distance.

  This class computes pairwise similarity between two BoxLists based on center L2 distance.
  """

  def _compare(self, boxlist1, boxlist2):
    """Compute pairwise center distance between the two BoxLists.

    Args:
      boxlist1: BoxList holding N boxes.
      boxlist2: BoxList holding M boxes.

    Returns:
      A tensor with shape [N, M] representing pairwise center distances.
    """

    ycenter1, xcenter1, _, _ = BoxList.get_center_coordinates_and_sizes(boxlist1)
    ycenter2, xcenter2, _, _ = BoxList.get_center_coordinates_and_sizes(boxlist2)

    centers1 = tf.transpose(tf.stack((ycenter1, xcenter1)))
    centers2 = tf.transpose(tf.stack((ycenter2, ycenter2)))

    centers_diff = tf.expand_dims(centers1, 1) - tf.expand_dims(centers2, 0)
    neg_l2_distance = -tf.norm(centers_diff, axis=2)
    return neg_l2_distance
    #return box_list_ops.iou(boxlist1, boxlist2)


center_sim = CenterDistanceSimilarity()
neg_l2_distance = center_sim.compare(gt_boxes, anchors)

# Indices of closest k anchors to each gt_box
top_k_anchors_per_gt = tf.math.top_k(neg_l2_distance, k=2)[1]

iou_selected_anchors = tf.gather(iou_values, top_k_anchors_per_gt, axis=1, batch_dims=1)

mean_iou_selected_anchors = tf.reduce_mean(iou_selected_anchors, axis=1)
std_iou_selected_anchors = tf.math.reduce_std(iou_selected_anchors, axis=1)

iou_thresholds = mean_iou_selected_anchors + std_iou_selected_anchors


def _set_values_using_indicator(x, indicator, val):
    """Set the indicated fields of x to val.

    Args:
      x: tensor.
      indicator: boolean with same shape as x.
      val: scalar with value to set.

    Returns:
      modified tensor.
    """
    indicator = tf.cast(indicator, x.dtype)
    return tf.add(tf.multiply(x, 1 - indicator), val * indicator)


# Matches for each column
# TODO: Remove not selected anchors based on distance

# top_k_anchor indices will be [[0, 1], [1, 2]]
# We need to create full indices like [[0, 0], [0, 1], [1, 2], [1, 1]]
range_rows = tf.expand_dims(tf.range(0, top_k_anchors_per_gt.get_shape()[0]), 1)  # will be [[0], [1]]
range_rows_repeated = tf.tile(range_rows, [1, top_k_anchors_per_gt.get_shape()[1]])  # will be [[0, 0], [1, 1]]
# change shapes to [N, k, 1] and [N, k, 1], to concatenate into [N, k, 2]
full_indices = tf.concat([tf.expand_dims(range_rows_repeated, -1), tf.expand_dims(top_k_anchors_per_gt, -1)], axis=2)
full_indices = tf.reshape(full_indices, [-1, 2])


# f = tf.reshape(top_k_anchors_per_gt, (4,1))
# r = tf.range(0, top_k_anchors_per_gt.shape[0])
# r = tf.transpose(tf.broadcast_to(r, top_k_anchors_per_gt.shape))
# r = tf.reshape(r, f.shape)
# indices = tf.concat((r, f), axis=1)

selected_anchors_by_distance = tf.cast(tf.scatter_nd(full_indices, tf.ones(tf.size(top_k_anchors_per_gt)), iou_values.shape), tf.int32)

selected_anchors_by_threshold = tf.cast(tf.greater_equal(iou_values, tf.expand_dims(iou_thresholds, 1)), tf.int32)

selected_anchors = selected_anchors_by_distance * selected_anchors_by_threshold

iou_values_positive_anchors = tf.cast(selected_anchors, tf.float32) * iou_values

mask_negative_anchors = tf.equal(tf.reduce_sum(selected_anchors, axis=0), 0)

matches = tf.argmax(iou_values_positive_anchors, 0, output_type=tf.int32)
matches = _set_values_using_indicator(matches, mask_negative_anchors, -1)


## TODO: Check anchors center inside gt
# selected_anchors_by_center
ycenter2, xcenter2, _, _ = BoxList.get_center_coordinates_and_sizes(anchors)
centers2 = tf.transpose(tf.stack((ycenter2, xcenter2)))



# Resultado final
# [ 0,  1,  1, -1] Matching de cada anchor a un gt o a ninguno (-1)

def point_outside_box(point, box):

    y_min, x_min, y_max, x_max = tf.split(
        value=boxlist.get(), num_or_size_splits=4, axis=1)
    win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)
    coordinate_violations = tf.concat([
        tf.less(y_min, win_y_min), tf.less(x_min, win_x_min),
        tf.greater(y_max, win_y_max), tf.greater(x_max, win_x_max)
    ], 1)
    valid_indices = tf.reshape(
        tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
    return gather(boxlist, valid_indices), valid_indices


#############################################
# Check if anchors' centers are in boxes area

ycenter2, xcenter2, _, _ = BoxList.get_center_coordinates_and_sizes(anchors)

gt_boxes_tensor = tf.convert_to_tensor(gt_boxes_array)
gt_boxes_broadcast_ymin = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 0), (gt_boxes_tensor.shape[0], 1)))
gt_boxes_broadcast_xmin = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 1), (gt_boxes_tensor.shape[0], 1)))
gt_boxes_broadcast_ymax = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 2), (gt_boxes_tensor.shape[0], 1)))
gt_boxes_broadcast_xmax = tf.squeeze(tf.slice(gt_boxes_tensor, (0, 3), (gt_boxes_tensor.shape[0], 1)))

is_in_xmin = tf.greater(xcenter2 - tf.transpose([gt_boxes_broadcast_xmin]), 0)
is_in_ymin = tf.greater(ycenter2 - tf.transpose([gt_boxes_broadcast_ymin]), 0)
is_in_xmax = tf.less(xcenter2 - tf.transpose([gt_boxes_broadcast_xmax]), 0)
is_in_ymax = tf.less(ycenter2 - tf.transpose([gt_boxes_broadcast_ymax]), 0)
selected_anchors_by_center_in_area = tf.logical_and(tf.logical_and(is_in_xmin, is_in_ymin), tf.logical_and(is_in_xmax, is_in_ymax))

# Mask similarly to selected_anchors_by_threshold or selected_anchors_by_distance
selected_anchors_by_center_in_area = tf.cast(selected_anchors_by_center_in_area, tf.int32)

#############################################


# Rest of the process
selected_anchors_by_distance = tf.cast(tf.scatter_nd(full_indices, tf.ones(tf.size(top_k_anchors_per_gt)), iou_values.shape), tf.int32)

selected_anchors_by_threshold = tf.cast(tf.greater_equal(iou_values, tf.expand_dims(iou_thresholds, 1)), tf.int32)

selected_anchors = selected_anchors_by_distance * selected_anchors_by_threshold * selected_anchors_by_center_in_area

iou_values_positive_anchors = tf.cast(selected_anchors, tf.float32) * iou_values

