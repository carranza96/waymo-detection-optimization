import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from waymo_open_dataset import dataset_pb2 as open_dataset
from utils_tf_record.read_dataset_utils import read_frame_waymo_segment, \
    read_and_parse_sharded_dataset, parse_camera_tfrecord_example, get_dataset_class_distribution
from waymo_open_dataset.protos import metrics_pb2, submission_pb2
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

tf.enable_eager_execution()

# FILENAME = "data/sample_rgb/training/training.record-00000-of-00798"
# FILENAME = "data/sample_rgb/training_eval.record"
# FILENAME = "data/camera_data/testing/testing.record-00000-of-00150"
# # FILENAME = "data/camera_frontal3/training/frontal_training.record-00000-of-00798"
# FILENAME = "data/camera_data/training/training.record-00010-of-00798"
#
# dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')

#FILENAME_PATTERN = "data/camera_data/training/*"
#FILENAME_PATTERN = "data/camera_data/training/training.record-0041?-of-00798"
#FILENAME_PATTERN = "data/camera_data/validation/validation.record-00000-of-00202"
FILENAME_PATTERN = "data/camera_frontal/training/frontal_training.record-00???-of-00798"

# FILENAME_PATTERN = "data/final_data/training/cyclists_training.record-00001-of-00003"
# FILENAME_PATTERN = "data/rgbRangeMedian/training/*"
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = True
dataset = tf.data.Dataset.list_files(FILENAME_PATTERN, shuffle=False, seed=20)
dataset = dataset.with_options(ignore_order)
dataset = dataset.interleave(tf.data.TFRecordDataset,
                             cycle_length=32,
                             num_parallel_calls=tf.data.experimental.AUTOTUNE)

# filename = "data/camera_lateral/training/lateral_training.record-00000-of-00798"
# dataset = tf.data.TFRecordDataset(filename, compression_type='')
# dataset = dataset.map(lambda data: parse_camera_tfrecord_example(data, False))

def plot_5cams():
    fig, ax = plt.subplots(1, 5, figsize=(20, 3))

    order = {
        'SIDE_LEFT': 0,
        'FRONT_LEFT': 1,
        'FRONT': 2,
        'FRONT_RIGHT': 3,
        'SIDE_RIGHT': 4}

    for i, raw_example in enumerate(dataset):

        if i < 5:
            parsed = tf.train.Example.FromString(raw_example.numpy())
            feature = parsed.features.feature

            if i % 1000 == 0:
                print(i)

            classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
            classes = feature['image/object/class/label'].int64_list.value

            source_id = feature['image/source_id'].bytes_list.value[0].decode()
            context_name = feature['image/context_name'].bytes_list.value[0].decode()
            frame = feature['image/frame_timestamp_micros'].int64_list.value[0]
            height = feature['image/height'].int64_list.value[0]
            width = feature['image/width'].int64_list.value[0]
            raw_img = feature['image/encoded'].bytes_list.value[0]

            image = tf.image.decode_jpeg(raw_img)

            xmins = [x * width for x in feature['image/object/bbox/xmin'].float_list.value]
            xmaxs = [x * width for x in feature['image/object/bbox/xmax'].float_list.value]
            ymins = [x * height for x in feature['image/object/bbox/ymin'].float_list.value]
            ymaxs = [x * height for x in feature['image/object/bbox/ymax'].float_list.value]

            # for z in range(len(xmins)):
            #     # if classes[z]==3:
            #         ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                            width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                            linewidth=1, edgecolor='red', facecolor='none'))

            camera_name = open_dataset.CameraName.Name.Name(feature["image/camera_name"].int64_list.value[0])
            ind = order[camera_name]
            ax[ind].imshow(image)

            ax[ind].set_title(camera_name, fontsize=16)
            ax[ind].grid(False)
            ax[ind].axis('off')
            # for line in [0.2, 0.4, 0.7]:
            #     ax[ind].axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')

            # break

    fig.savefig('fig3.png', format='png', dpi=1200, bbox_inches='tight')
    plt.show()


def plot_bboxes_levels():
    colors = {1: 'red', 2: 'blue', 3: 'yellow'}

    dif_colors = {0: 'green', 2: 'red'}

    for i, raw_example in enumerate(dataset.skip(1000)):
        #
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature
        if i % 1000 == 0:
            print(i)

        classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
        classes = feature['image/object/class/label'].int64_list.value

        source_id = feature['image/source_id'].bytes_list.value[0].decode()
        context_name = feature['image/context_name'].bytes_list.value[0].decode()
        frame = feature['image/frame_timestamp_micros'].int64_list.value[0]
        height = feature['image/height'].int64_list.value[0]
        width = feature['image/width'].int64_list.value[0]
        raw_img = feature['image/encoded'].bytes_list.value[0]

        xmins = [x * width for x in feature['image/object/bbox/xmin'].float_list.value]
        xmaxs = [x * width for x in feature['image/object/bbox/xmax'].float_list.value]
        ymins = [x * height for x in feature['image/object/bbox/ymin'].float_list.value]
        ymaxs = [x * height for x in feature['image/object/bbox/ymax'].float_list.value]

        widths = [xmax - xmin for (xmin, xmax) in zip(xmins, xmaxs)]

        difficulties = feature['image/object/difficult'].int64_list.value

        ind_bikes = [i for i in range(len(classes)) if classes[i] == 3]
        big_bikes = [i for i in ind_bikes if ymins[i] > 400 and widths[i] > 100]

        # if len(np.unique(classes)) == 3 and len(big_bikes)>0 and 2 in difficulties:
        if 2 in difficulties and len(np.unique(classes)) >= 2:

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)
            image = tf.image.decode_jpeg(raw_img)

            for z in range(len(xmins)):
                # if classes[z]==3:
                ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
                                               width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
                                               linewidth=1, edgecolor=colors[classes[z]], facecolor='none'))

            # camera_name = open_dataset.CameraName.Name.Name(feature["image/camera_name"].int64_list.value[0])
            # ind = order[camera_name]
            ax.imshow(image)

            # ax.set_title(camera_name, fontsize=14)
            ax.grid(False)
            ax.axis('off')
            # ['Vehicle', 'Pedestrian', 'Cyclist']
            # legend1 = ax.legend(*fig.legend_elements(),
            #                        loc="best", title="Classes")
            red_patch = patches.Patch(color='red', label='Vehicle')
            blue_patch = patches.Patch(color='blue', label='Pedestrian')
            yellow_patch = patches.Patch(color='yellow', label='Cyclist')
            ax.legend(handles=[red_patch, blue_patch, yellow_patch], loc="best", title="Classes", title_fontsize=10,
                      fontsize=10)
            # ax.add_artist(legend1)
            # for line in [0.2, 0.4, 0.7]:
            #     ax[ind].axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')
            plt.show()
            # break
            # fig.savefig('bboxes.png', format='png', dpi=1200, bbox_inches='tight')
            print()

            fig2 = plt.figure()
            ax = plt.subplot(1, 1, 1)
            image = tf.image.decode_jpeg(raw_img)

            for z in range(len(xmins)):
                # if classes[z]==3:
                ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
                                               width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
                                               linewidth=1, edgecolor=dif_colors[difficulties[z]], facecolor='none'))

            # camera_name = open_dataset.CameraName.Name.Name(feature["image/camera_name"].int64_list.value[0])
            # ind = order[camera_name]
            ax.imshow(image)

            # ax.set_title(camera_name, fontsize=14)
            ax.grid(False)
            ax.axis('off')
            # ['Vehicle', 'Pedestrian', 'Cyclist']
            # legend1 = ax.legend(*fig.legend_elements(),
            #                        loc="best", title="Classes")
            green_patch = patches.Patch(color='green', label='Level 1')
            red_patch = patches.Patch(color='red', label='Level 2')
            ax.legend(handles=[green_patch, red_patch], loc="best", title="Difficulty levels", title_fontsize=10,
                      fontsize=10)
            # ax.add_artist(legend1)
            # for line in [0.2, 0.4, 0.7]:
            #     ax[ind].axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')
            plt.show()
            # break
            # fig2.savefig('bboxes_diff.png', format='png', dpi=1200, bbox_inches='tight')
            print()

    # break





def plot_regions_over_images():
    for i, raw_example in enumerate(dataset):
        #
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature
        if i % 1000 == 0:
            print(i)

        classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
        classes = feature['image/object/class/label'].int64_list.value

        source_id = feature['image/source_id'].bytes_list.value[0].decode()
        context_name = feature['image/context_name'].bytes_list.value[0].decode()
        frame = feature['image/frame_timestamp_micros'].int64_list.value[0]
        height = feature['image/height'].int64_list.value[0]
        width = feature['image/width'].int64_list.value[0]
        raw_img = feature['image/encoded'].bytes_list.value[0]
        image = tf.image.decode_jpeg(raw_img)

        xmins = [x * width for x in feature['image/object/bbox/xmin'].float_list.value]
        xmaxs = [x * width for x in feature['image/object/bbox/xmax'].float_list.value]
        ymins = [x * height for x in feature['image/object/bbox/ymin'].float_list.value]
        ymaxs = [x * height for x in feature['image/object/bbox/ymax'].float_list.value]

        widths = [xmax - xmin for (xmin, xmax) in zip(xmins, xmaxs)]

        difficulties = feature['image/object/difficult'].int64_list.value


        if any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.2)\
                and any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.4 and ymaxs if y>image.get_shape().as_list()[0] * 0.2)\
                and any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.7 and ymaxs if y>image.get_shape().as_list()[0] * 0.4)\
                and any(y for y in ymaxs if y>image.get_shape().as_list()[0] * 0.7):

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)

            for line in [0.2, 0.4, 0.7]:
                ax.axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')

            for z in range(0,len(xmins),4):

                # if classes[z]==3:
                ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
                                               width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
                                               linewidth=1, edgecolor='green', facecolor='none'))
            # camera_name = open_dataset.CameraName.Name.Name(feature["image/camera_name"].int64_list.value[0])
            # ind = order[camera_name]
            ax.imshow(image)

            # ax.set_title(camera_name, fontsize=14)
            ax.grid(False)
            ax.axis('off')
            # ['Vehicle', 'Pedestrian', 'Cyclist']
            # legend1 = ax.legend(*fig.legend_elements(),
            #                        loc="best", title="Classes")
            props = dict(boxstyle='round', facecolor="white")

            # place a text box in upper left in axes coords
            ax.text(0.95, 0.95, 'R1', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            ax.text(0.95, 0.75, 'R2', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            ax.text(0.95, 0.55, 'R3', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
            ax.text(0.95, 0.25, 'R4', transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)


            # red_patch = patches.Patch(color='red', label='Vehicle')
            # blue_patch = patches.Patch(color='blue', label='Pedestrian')
            # yellow_patch = patches.Patch(color='yellow', label='Cyclist')
            # ax.legend(handles=[red_patch, blue_patch, yellow_patch], loc="best", title="Classes", title_fontsize=10,
            #           fontsize=10)
            # ax.add_artist(legend1)
            # for line in [0.2, 0.4, 0.7]:
            #     ax[ind].axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')
            plt.show()
            # break
            fig.savefig('regions_image.png', format='png', dpi=1200, bbox_inches='tight')
            print()



def get_region(ycenter):
    r = 0
    if ycenter>0.2 and ycenter <0.4:
        r=1
    if ycenter > 0.4 and ycenter < 0.7:
        r=2
    if ycenter>0.7:
        r=3

def generate_anchors(ratios=None, scales=None, base_size=256):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales w.r.t. a reference window.
    """
    num_anchors = len(ratios) * len(scales)
    # initialize output anchors
    anchors = np.zeros((num_anchors, 4))
    # scale base_size
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T
    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]
    # correct for ratios
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))
    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T
    return anchors


def plot_anchor_opt():
    for i, raw_example in enumerate(dataset):
        #
        parsed = tf.train.Example.FromString(raw_example.numpy())
        feature = parsed.features.feature
        if i % 1000 == 0:
            print(i)

        classes_text = [x.decode() for x in feature['image/object/class/text'].bytes_list.value]
        classes = feature['image/object/class/label'].int64_list.value

        source_id = feature['image/source_id'].bytes_list.value[0].decode()
        context_name = feature['image/context_name'].bytes_list.value[0].decode()
        frame = feature['image/frame_timestamp_micros'].int64_list.value[0]
        height = feature['image/height'].int64_list.value[0]
        width = feature['image/width'].int64_list.value[0]
        raw_img = feature['image/encoded'].bytes_list.value[0]
        image = tf.image.decode_jpeg(raw_img)

        xmins = [x * width for x in feature['image/object/bbox/xmin'].float_list.value]
        xmaxs = [x * width for x in feature['image/object/bbox/xmax'].float_list.value]
        ymins = [x * height for x in feature['image/object/bbox/ymin'].float_list.value]
        ymaxs = [x * height for x in feature['image/object/bbox/ymax'].float_list.value]

        widths = [xmax - xmin for (xmin, xmax) in zip(xmins, xmaxs)]

        difficulties = feature['image/object/difficult'].int64_list.value
        # and any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.4 and ymaxs if y>image.get_shape().as_list()[0] * 0.2)\
        # and any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.7 and ymaxs if y>image.get_shape().as_list()[0] * 0.4)\

        if any(y for y in ymaxs if y<image.get_shape().as_list()[0] * 0.2)\
                and any(y for y in ymaxs if y>image.get_shape().as_list()[0] * 0.7):
        #if True:

            fig = plt.figure()
            ax = plt.subplot(1, 1, 1)

            # for line in [0.2, 0.4, 0.7]:
            #     ax.axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--',xmin=0.9)

            # for z in [1, 4, 6]:
            # # if classes[z]==3:
            #     ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                linewidth=3, edgecolor='green', facecolor='none'))

            # for z in [1]:
            #
            #
            #     # if classes[z]==3:
            #     ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                    width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                    linewidth=3, edgecolor='green', facecolor='none'))
            #     ycenter = ymaxs[z] - (ymaxs[z] - ymins[z]) / 2
            #     xcenter = xmaxs[z] - (xmaxs[z] - xmins[z]) / 2
            #
            #     anchors = generate_anchors(ratios=[2], scales=[1, 2]) # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='red', facecolor='none'))
            #
            #     anchors = generate_anchors(ratios=[2], scales=[0.25,0.5])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,linestyle='--',
            #                                        linewidth=3, edgecolor='red', facecolor='none'))
            #
            #     anchors = generate_anchors(ratios=[2.1], scales=[0.693, 1.254, 1.935, 2.5])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='blue', facecolor='none'))

            #     for z in [3]:
            #
            #         # if classes[z]==3:
            #         ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                        width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                        linewidth=1, edgecolor='green', facecolor='none'))
            #         ycenter = ymaxs[z] - (ymaxs[z] - ymins[z]) / 2
            #
            #         anchors = generate_anchors(ratios=[1], scales=[2, 1])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #         for anchor in anchors:
            #             anchor_height = anchor[2] - anchor[0]
            #             anchor_width = anchor[3] - anchor[1]
            #             ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                            width=anchor_width, height=anchor_height,
            #                                            linewidth=1, edgecolor='red', facecolor='none'))
            #         anchors = generate_anchors(ratios=[0.905], scales=[1.573])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #         for anchor in anchors:
            #             anchor_height = anchor[2] - anchor[0]
            #             anchor_width = anchor[3] - anchor[1]
            #             ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                            width=anchor_width, height=anchor_height,
            #                                            linewidth=1, edgecolor='blue', facecolor='none'))
            #

            # for z in [6]:
            #     # if classes[z]==3:
            #     ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                    width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                    linewidth=3, edgecolor='green', facecolor='none'))
            #     ycenter = ymaxs[z] - (ymaxs[z] - ymins[z]) / 2
            #     xcenter = xmaxs[z] - (xmaxs[z] - xmins[z]) / 2
            #
            #     anchors = generate_anchors(ratios=[2], scales=[0.25, 0.5])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='red', facecolor='none'))
            #     #
            #     anchors = generate_anchors(ratios=[2], scales=[1, 2])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='red', facecolor='none', linestyle='--'))
            #
            #     anchors = generate_anchors(ratios=[2.2], scales=[0.074, 0.158, 0.25, 0.414])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='blue', facecolor='none'))
            #

            # for z in [4]:
            #
            #     # if classes[z]==3:
            #     ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
            #                                    width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
            #                                    linewidth=3, edgecolor='green', facecolor='none'))
            #     ycenter = ymaxs[z] - (ymaxs[z] - ymins[z]) / 2
            #     xcenter = xmaxs[z] - (xmaxs[z] - xmins[z]) / 2
            #
            #
            #     anchors = generate_anchors(ratios=[1], scales=[0.25, 0.5])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='red', facecolor='none'))
            #     #
            #     anchors = generate_anchors(ratios=[1], scales=[1, 2])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='red', facecolor='none', linestyle='--'))
            #
            #     anchors = generate_anchors(ratios=[0.905], scales=[0.082, 0.155, 0.254, 0.500])  # 'ymin', 'xmin', 'ymax', 'xmax'
            #
            #     for anchor in anchors:
            #         anchor_height = anchor[2] - anchor[0]
            #         anchor_width = anchor[3] - anchor[1]
            #         ax.add_patch(patches.Rectangle(xy=(xcenter - anchor_width/2, ycenter - anchor_height/2),
            #                                        width=anchor_width, height=anchor_height,
            #                                        linewidth=3, edgecolor='blue', facecolor='none'))




        # camera_name = open_dataset.CameraName.Name.Name(feature["image/camera_name"].int64_list.value[0])
            # ind = order[camera_name]
            #ax.imshow(image)

            # ax.set_title(camera_name, fontsize=14)
            ax.grid(False)
            ax.axis('off')
            # ['Vehicle', 'Pedestrian', 'Cyclist']
            # legend1 = ax.legend(*fig.legend_elements(),
            #                        loc="best", title="Classes")
            props = dict(boxstyle='round', facecolor="white")

            # place a text box in upper left in axes coords
            # ax.text(0.95, 0.95, 'R1', transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', bbox=props)
            # ax.text(0.95, 0.75, 'R2', transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', bbox=props)
            # ax.text(0.95, 0.55, 'R3', transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', bbox=props)
            # ax.text(0.95, 0.25, 'R4', transform=ax.transAxes, fontsize=10,
            #         verticalalignment='top', bbox=props)

            green_patch = patches.Patch(color='green', label='Ground truth')
            red_patch = patches.Patch(color='red', label='Default anchors')
            blue_patch = patches.Patch(color='blue', label='Optimized anchors')
            ax.legend(handles=[green_patch, red_patch, blue_patch], loc="upper center", bbox_to_anchor=(0.5, 0.5), ncol=1, fontsize=10)

        # red_patch = patches.Patch(color='red', label='Vehicle')
            # blue_patch = patches.Patch(color='blue', label='Pedestrian')
            # yellow_patch = patches.Patch(color='yellow', label='Cyclist')
            # ax.legend(handles=[red_patch, blue_patch, yellow_patch], loc="best", title="Classes", title_fontsize=10,
            #           fontsize=10)
            # ax.add_artist(legend1)
            # for line in [0.2, 0.4, 0.7]:
            #     ax[ind].axhline(y=image.get_shape().as_list()[0]*line, color='r', linestyle='--')
            plt.show()
            # break
            fig.savefig('regions_image_legend.png', format='png', dpi=1200, bbox_inches='tight')
            print()

# plot_bboxes_levels()
plot_anchor_opt()
