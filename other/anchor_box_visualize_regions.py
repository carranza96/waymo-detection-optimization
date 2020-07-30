import tensorflow as tf
from object_detection.anchor_generators.regions_grid_anchor_generator import RegionsGridAnchorGenerator
from object_detection.anchor_generators.grid_anchor_generator import GridAnchorGenerator
from object_detection.models.faster_rcnn_resnet_v1_feature_extractor_test import FasterRcnnResnetV1FeatureExtractorTest
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
from object_detection.core import box_list_ops
import time

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
    anchor_generator = RegionsGridAnchorGenerator(**anchor_kwargs)

    anchor_box_lists = anchor_generator.generate(feature_map_shape_list)

    anchor_box_lists = [box_list_ops.to_normalized_coordinates(anchor_box_lists[0], 1280, 1920, check_range=False)]

    feature_map_boxes = {}

    with tf.Session() as sess:
        for shape, box_list in zip(feature_map_shape_list, anchor_box_lists):
            feature_map_boxes[shape] = sess.run(box_list.data['boxes'])

    return feature_map_boxes


def draw_boxes(boxes, figsize, nrows, ncols, grid=(0, 0), regions_limits=[]):
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for ax, box in zip(axes.flat, boxes):
        ymin, xmin, ymax, xmax = box
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=True, facecolor=(1,0,0,0.2), edgecolor=(1,0,0,1), lw=2))
        # add region limits
        for region_lim in regions_limits:
            ax.axhline(y=region_lim, c='blue')
        # add gridlines to represent feature map cells
        ax.set_xticks(np.linspace(0, 1, grid[0] + 1), minor=True)
        ax.set_yticks(np.linspace(0, 1, grid[1] + 1), minor=True)
        ax.invert_yaxis()
        ax.grid(True, which='minor', axis='both')

    fig.tight_layout()

    return fig

feature_map_shape_list = get_feature_map_shapes(1280, 1920)

regions_limits = [0.17, 0.4, 0.7]
boxes = get_feature_map_anchor_boxes(
    regions_limits=regions_limits,
    scales=[[0.08, 0.21, 0.47], [0.09, 0.26, 0.59], [0.10, 0.37, 0.97], [1.20, 1.80, 2.46]],
    aspect_ratios=[[1, 1.89, 3.35], [0.63, 1.49, 2.81], [0.48, 1.28, 2.50], [0.90, 2.07, 3.95]],
    base_anchor_size=None,
    anchor_stride=None,
    anchor_offset=None,
    special_cases=[# x_position, y_position, anchor_index, scale, aspect_ratio
                    (0.5, 0.5, 0, 6.1, 2/3), # center
                    (0.25, 0.5, 0, 4.3, 4/3), # left
                    (0.75, 0.5, 0, 4.3, 4/3), # right
                    (0.5, 0.25, 0, 4.3, 1/3),  # top
                    (0.5, 0.75, 0, 4.3, 1/3),  # bottom
                  ],
    feature_map_shape_list=feature_map_shape_list
)

n_boxes = boxes[(80, 120)].shape[0]

regions = [(0 if i == 0 else regions_limits[i-1],
            1 if i == len(regions_limits) else regions_limits[i])
           for i in range(len(regions_limits)+1)]

for lims in regions:
    fig = draw_boxes(boxes[(80, 120)][int(n_boxes*np.mean(lims)):], figsize=(60, 15), nrows=3, ncols=9, grid=(80, 120), regions_limits=regions_limits)
    fig.show()

boxes_to_plot = []
for lims in regions:
    boxes_to_plot += boxes[(80, 120)][int(n_boxes*np.mean(lims)):int(n_boxes*np.mean(lims))+9].tolist()
fig = draw_boxes(boxes_to_plot, figsize=(60, len(regions)*int(15/3)), nrows=len(regions), ncols=9, grid=(80, 120), regions_limits=regions_limits)
fig.show()

boxes_to_plot = [boxes[(80, 120)][int(22140)],
                 boxes[(80, 120)][int(43470)],
                 boxes[(80, 120)][int(43740)],
                 boxes[(80, 120)][int(44010)],
                 boxes[(80, 120)][int(65340)]]
fig = draw_boxes(boxes_to_plot, figsize=(int(5 * 60/9), int(1 * 15/3)), nrows=1, ncols=5, grid=(80, 120), regions_limits=regions_limits)
fig.show()

t0 = time.time()

anchor_generator = GridAnchorGenerator(scales=(0.25, 0.5, 1.0, 2.0),
                                           aspect_ratios=(0.5, 1.0, 2.0),
                                           base_anchor_size=None,
                                           anchor_stride=None,
                                           anchor_offset=None,)
for i in range(10):
    anchor_box_lists = anchor_generator.generate(feature_map_shape_list)

t1 = time.time()

anchor_generator = RegionsGridAnchorGenerator(regions_limits=regions_limits,
                                                  scales=[[0.08, 0.21, 0.47], [0.09, 0.26, 0.59], [0.10, 0.37, 0.97], [1.20, 1.80, 2.46]],
                                                  aspect_ratios=[[1, 1.89, 3.35], [0.63, 1.49, 2.81], [0.48, 1.28, 2.50], [0.90, 2.07, 3.95]],
                                                  base_anchor_size=None,
                                                  anchor_stride=None,
                                                  anchor_offset=None,
                                                  special_cases=[# x_position, y_position, anchor_index, scale, aspect_ratio
                                                                  (0.5, 0.5, 0, 6.1, 2/3), # center
                                                                  (0.25, 0.5, 0, 4.3, 4/3), # left
                                                                  (0.75, 0.5, 0, 4.3, 4/3), # right
                                                                  (0.5, 0.25, 0, 4.3, 1/3),  # top
                                                                  (0.5, 0.75, 0, 4.3, 1/3),  # bottom
                                                                ])
for i in range(10):
    anchor_box_lists = anchor_generator.generate(feature_map_shape_list)

t2 = time.time()

print("######### TIME ########")
print("# Grid AG: {:.2f}s".format((t1-t0)/10))
print("# Regions AG: {:.2f}s".format((t2-t1)/10))
print("#######################")