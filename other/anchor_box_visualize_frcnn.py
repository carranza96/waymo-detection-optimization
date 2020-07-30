import tensorflow as tf
from object_detection.anchor_generators.grid_anchor_generator import GridAnchorGenerator
from object_detection.models.faster_rcnn_resnet_v1_feature_extractor_test import FasterRcnnResnetV1FeatureExtractorTest
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
from object_detection.core import box_list_ops
from object_detection.core import box_list

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_feature_map_shapes(image_height, image_width):
    """
    :param image_height: height in pixels
    :param image_width: width in pixels
    :returns: list of tuples containing feature map resolutions
    """

    feature_extractor = FasterRcnnResnetV1FeatureExtractorTest()._build_feature_extractor(
                               first_stage_features_stride=16,
                               activation_fn=tf.nn.relu,
                               architecture='resnet_v1_101'
    )

    image_batch_tensor = tf.zeros([1, image_height, image_width, 1])
    rpn_feature_map, _ = feature_extractor.extract_proposal_features(
        image_batch_tensor, scope='TestScope')
    return [tuple(rpn_feature_map.get_shape().as_list()[1:3])]


def get_feature_map_anchor_boxes(feature_map_shape_list, **anchor_kwargs):
    """
    :param feature_map_shape_list: list of tuples containing feature map resolutions
    :returns: dict with feature map shape tuple as key and list of [ymin, xmin, ymax, xmax] box co-ordinates
    """
    anchor_generator = GridAnchorGenerator(**anchor_kwargs)

    anchors_boxlist= anchor_generator.generate(feature_map_shape_list)

    # anchor_box_lists = [box_list_ops.to_normalized_coordinates(anchor_box_lists[0], 1280, 1920, check_range=False)]
    clip_window = tf.cast(tf.stack([0, 0, 1280, 1920]),
                          dtype=tf.float32)

    feature_map_boxes = {}

    with tf.Session() as sess:
        for shape, box_list in zip(feature_map_shape_list, anchors_boxlist):
            # box_list = box_list_ops.clip_to_window(
            #     box_list, clip_window, filter_nonoverlapping=False)
            feature_map_boxes[shape] = sess.run(box_list.data['boxes'])

    return feature_map_boxes


def draw_boxes(boxes, figsize, nrows, ncols, grid=(0, 0)):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, box in zip(axes.flat, boxes):
        ymin, xmin, ymax, xmax = box
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, edgecolor='red', lw=2))

        # add gridlines to represent feature map cells
        ax.set_xticks(np.linspace(0, 1, grid[0] + 1), minor=True)
        ax.set_yticks(np.linspace(0, 1, grid[1] + 1), minor=True)
        ax.grid(True, which='minor', axis='both')

    fig.tight_layout()

    return fig

# print(get_feature_map_shapes(1280, 1920))
boxes = get_feature_map_anchor_boxes(
    scales=(0.25, 0.5, 1.0, 2.0),
    aspect_ratios=(0.5, 1.0, 2.0),
    base_anchor_size=None,
    anchor_stride=None,
    anchor_offset=None,
    feature_map_shape_list=get_feature_map_shapes(1280, 1920)
)



# fig = draw_boxes(boxes[(80, 120)][50000:], figsize=(60, 15), nrows=3, ncols=9, grid=(80, 120))
# fig.show()