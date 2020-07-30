import numpy as np
import tensorflow as tf
from object_detection.utils import dataset_util
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import label_pb2 as open_dataset_labels
from time import time
from scipy.ndimage import maximum_filter, median_filter
from imageio import imread
from multiprocessing import Pool
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_tf_example(frame_context, camera_context, camera_image, camera_labels,
                      timestamp_micros, channels=3, additional_channels=None, ignore_difficult_instances=False):
    height = camera_context.height
    width = camera_context.width
    encoded_image_data = camera_image.image  # Encoded image bytes
    image_format = b'jpeg'
    segment_id = frame_context.name.encode()
    image_camera_name = camera_image.name
    source_id = segment_id + "_".encode() + str(image_camera_name).encode() + "_".encode() \
                + str(timestamp_micros).encode()
    time_of_day = frame_context.stats.time_of_day.encode()
    location = frame_context.stats.location.encode()
    weather = frame_context.stats.weather.encode()


    # TODO: Change bbox features to bytes and change reading in obj detection API
    # TODO: https://stackoverflow.com/questions/40184812/tensorflow-is-it-possible-to-store-tf-record-sequence-examples-as-float16
    xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
    ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
    classes_text = []  # List of string class name of bounding box (1 per box)
    classes = []  # List of integer class id of bounding box (1 per box)
    difficulty_levels = []  # Level of difficulty of labels (1 per box)

    if camera_labels:
        for label in camera_labels.labels:
            # Difficulty can be 0 or 2
            difficulty = label.detection_difficulty_level
            if ignore_difficult_instances and difficulty:
                continue

            # Normalized coordinates
            xmins.append((label.box.center_x - label.box.length / 2) / width)
            xmaxs.append((label.box.center_x + label.box.length / 2) / width)
            ymins.append((label.box.center_y - label.box.width / 2) / height)
            ymaxs.append((label.box.center_y + label.box.width / 2) / height)
            label_name = open_dataset_labels.Label.Type.Name(label.type)
            classes_text.append(label_name.encode('utf-8'))
            # Change label of cyclist from 4 to 3: Only 3 classes for 2D detection (Vehicle, Pedestrian, Cyclist)
            label_id = 3 if label_name == "TYPE_CYCLIST" else label.type
            classes.append(label_id)
            difficulty_levels.append(difficulty)

    contains_cyclist = 3 in classes
    contains_labels = len(classes) > 0
        # if source_id.decode() == '1457696187335927618_595_027_615_027_1_1520902837711487':
        #     print(label.box.length, label.box.width)
        # print([ymax-ymin for ymax, ymin in zip(ymaxs, ymins)])
        #
        # print("Found")
    fig = plt.figure()
    ax = plt.subplot(1, 1, 1)
    image = tf.image.decode_jpeg(encoded_image_data)


    plt.imshow(image)

    xmins = [x * width for x in xmins]
    xmaxs = [x * width for x in xmaxs]
    ymins = [x * height for x in ymins]
    ymaxs = [x * height for x in ymaxs]
    dif_colors = {0: 'green', 2: 'red'}
    for z in range(len(xmins)):
        # if classes[z]==3:
        ax.add_patch(patches.Rectangle(xy=(xmins[z], ymins[z]),
                                       width=xmaxs[z] - xmins[z], height=ymaxs[z] - ymins[z],
                                       linewidth=1, edgecolor=dif_colors[difficulty_levels[z]], facecolor='none'))

    green_patch = patches.Patch(color='green', label='Level 1')
    red_patch = patches.Patch(color='red', label='Level 2')
    ax.legend(handles=[green_patch, red_patch], loc="best", title="Difficulty levels", title_fontsize=10,
              fontsize=10)
    plt.show()

    feature = {
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/channels': dataset_util.int64_feature(channels),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/source_id': dataset_util.bytes_feature(source_id),
        'image/context_name': dataset_util.bytes_feature(segment_id),
        'image/frame_timestamp_micros': dataset_util.int64_feature(timestamp_micros),
        'image/camera_name': dataset_util.int64_feature(image_camera_name),
        'image/time_of_day': dataset_util.bytes_feature(time_of_day),
        'image/location': dataset_util.bytes_feature(location),
        'image/weather': dataset_util.bytes_feature(weather),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficulty_levels)
    }

    if additional_channels:
        feature['image/additional_channels/encoded'] = dataset_util.bytes_feature(additional_channels)


    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))

    return tf_example, contains_labels, contains_cyclist


def preprocess_rgb(frame, frame_index, frames_to_skip, ignore_difficult_instances=False, is_testing=False):
    """ Standard preprocessing """
    tf_examples = []

    for image_index in range(len(frame.images)):
        camera_image = frame.images[image_index]

        camera_labels = []
        if frame.camera_labels:
            camera_labels = frame.camera_labels[image_index]

        # Search for camera[i] in frame.context.camera_calibrations since it does not follow
        # same order as camera_image and camera_labels
        camera_context = next(context for context in frame.context.camera_calibrations
                              if context.name == camera_image.name)

        if camera_context.height != 886:
            continue

        tf_example, contains_labels, contains_cyclist = create_tf_example(frame.context, camera_context, camera_image,
                                       camera_labels, frame.timestamp_micros, channels=3,
                                       ignore_difficult_instances=ignore_difficult_instances)

        if (frame_index % frames_to_skip == 0 and contains_labels) or contains_cyclist or is_testing:
            tf_examples.append(tf_example)

    return tf_examples





## TODO: Study best sky value
## TODO: Use other lidars, not only TOP LIDAR
## TODO: Use second return from lidar
## TODO: Use other lidar measure (range, intensity, elongation)
LIDAR = 1

SKY_VALUE = {
    0: 0,
    1: 0,
    2: 0
}

LIDAR_MEASURE = {
    0: 'Range',
    1: 'Intensity',
    2: 'Elongation'
}

MAX_LIDAR_MEASURES = {
    0: 74.99496 + 1,
    1: 1,
    2: 1.493107 + 0.1
}


NORMALIZATION = {
    0: lambda x: 255 * (MAX_LIDAR_MEASURES[0] - x)/MAX_LIDAR_MEASURES[0],
    1: lambda x: 255 * np.tanh(x),
    2: lambda x: 255 * x/MAX_LIDAR_MEASURES[2]
}

KERNEL = (22, 11)


def _process_image_lidargb(image_index, frame, lidar_projections, max_val, ignore_difficult_instances):

    camera_image = frame.images[image_index]
    image_matrix = imread(camera_image.image)
    # Search for camera[i] in frame.context.camera_calibrations since it does not follow
    # same order as camera_image and camera_labels
    ## TODO: Order of camera_images and lidar projections is different
    camera_context = next(context for context in frame.context.camera_calibrations
                          if context.name == camera_image.name)

    # Filter projections corresponding to this camera
    mask = np.equal(lidar_projections[:, 1], camera_context.name)
    matching_projections = lidar_projections[mask]

    # TODO: Remove matching projections from global tensor lidar projections
    # Get y, x indices from channels 2,3 and reverse
    indices = np.flip(matching_projections[:, -2:], [1]).astype(int)

    # Get values
    values = matching_projections[:, 0]
    #Normalize according to max values
    values = np.round(255 * values / max_val).astype(np.int32)

    # Create LiDAR channel
    lidar_projection_channel = SKY_VALUE[0] * np.ones(image_matrix.shape[:2])
    lidar_projection_channel[indices[:, 0], indices[:, 1]] = values

    # lidar_projection_channel = lidar_projection_channel.astype(np.uint8)
    lidar_projection_channel_shape = lidar_projection_channel.shape

    # Local maximum filter
    lidar_projection_channel = -1 * maximum_filter(-1 * lidar_projection_channel, size=(22, 11), mode="reflect")
    lidar_projection_channel = lidar_projection_channel.astype(np.uint8)
    lidar_projection_channel = np.reshape(lidar_projection_channel, lidar_projection_channel_shape)

    # Replace R channel with LiDAR channel
    # TODO: Can this below be done faster?
    g_channel = np.reshape(image_matrix[:, :, 1], (image_matrix.shape[0], image_matrix.shape[1]))
    b_channel = np.reshape(image_matrix[:, :, 2], (image_matrix.shape[0], image_matrix.shape[1]))
    image_matrix = np.stack([lidar_projection_channel, g_channel, b_channel], axis=-1)

    ## TODO: If change object deteciton api: store as encoded numpy string instead of jpeg (maybe not necessary)
    camera_image.image = tf.image.encode_jpeg(image_matrix, quality=89, optimize_size=True).numpy()
    channels = image_matrix.shape[2]

    camera_labels = frame.camera_labels[image_index]

    tf_example = create_tf_example(frame_context=frame.context,
                                   camera_context=camera_context,
                                   camera_image=camera_image,
                                   camera_labels=camera_labels,
                                   timestamp_micros=frame.timestamp_micros,
                                   channels=channels,
                                   ignore_difficult_instances=ignore_difficult_instances)
    return tf_example


# TODO: Esta mal, hay que meterle la segunda proyeccion
def preprocess_lidar_gb(frame, ignore_difficult_instances):
    """ Change red channel for LiDAR projection """
    tf_examples = []

    (range_images, camera_projections, _) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    # Range image
    range_image = range_images[LIDAR][0]
    range_image_numpy = np.asarray(range_image.data)
    range_image_numpy = np.reshape(range_image_numpy, range_image.shape.dims)
    # TODO: Use other lidars as well (here channel 0: range)
    range_image_numpy = range_image_numpy[..., 0]
    # Change shape to (width,height,1) and cast to int
    range_image_numpy = np.expand_dims(range_image_numpy, axis=2).astype(np.int32)

    # TODO: Round to closest integer?
    max_val = tf.cast(MAX_LIDAR_MEASURES[0], tf.int32)

    # Camera projection
    camera_projection = camera_projections[LIDAR][0]
    camera_projection_numpy = np.asarray(camera_projection.data)
    camera_projection_numpy = np.reshape(camera_projection_numpy, camera_projection.shape.dims)
    # TODO: Use second projection (Here only use first projection channels 0,1,2)
    camera_projection_numpy = camera_projection_numpy[..., :3]

    # Concat lidar and camera projection
    lidar_projection = np.concatenate([range_image_numpy, camera_projection_numpy], axis=2)
    # Flatten rows and columns. Resulting shape (169600,4)
    lidar_projections = np.reshape(lidar_projection, [lidar_projection.shape[0] * lidar_projection.shape[1], -1])

    # for image_index in range(len(frame.images)):
    for image_index in range(len(frame.images)):
        tf_examples.append(_process_image_lidargb(image_index, frame, lidar_projections, max_val, ignore_difficult_instances))

    return tf_examples


def preprocess_rgb_range(frame, frame_index, frames_to_skip, ignore_difficult_instances=False, is_testing=False):
    """ Change red channel for LiDAR projection """
    tf_examples = []

    (range_images, camera_projections, _) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    # Range image
    range_image = range_images[LIDAR][0]
    range_image_numpy = np.asarray(range_image.data)
    range_image_numpy = np.reshape(range_image_numpy, range_image.shape.dims)
    range_image_numpy = range_image_numpy[..., 0]
    # Change shape to (width,height,1) and cast to int
    range_image_numpy = np.expand_dims(range_image_numpy, axis=2).astype(np.int32)

    # max_val = tf.cast(MAX_LIDAR_MEASURES[0], tf.int32)

    # Camera projection
    camera_projection = camera_projections[LIDAR][0]
    camera_projection_numpy = np.asarray(camera_projection.data)
    camera_projection_numpy = np.reshape(camera_projection_numpy, camera_projection.shape.dims)
    camera_projection1_numpy = camera_projection_numpy[..., :3]
    camera_projection2_numpy = camera_projection_numpy[..., 3:]

    # Concat lidar and camera projection (Shape: (64,2650,6))
    # Description of 4 channels in lidar_projection1/2:
    # 0: range, 1: camera, 2: x position, 3: y position
    lidar_projection1 = np.concatenate([range_image_numpy, camera_projection1_numpy], axis=2)
    lidar_projection2 = np.concatenate([range_image_numpy, camera_projection2_numpy], axis=2)

    # Flatten rows and columns. Resulting shape (169600,6)
    lidar_projection1 = np.reshape(lidar_projection1, [lidar_projection1.shape[0] * lidar_projection1.shape[1], -1])
    lidar_projection2 = np.reshape(lidar_projection2, [lidar_projection2.shape[0] * lidar_projection2.shape[1], -1])

    # Concat first and second projections in one array of shape (339200, 6)
    lidar_projections = np.concatenate([lidar_projection1, lidar_projection2])

    t = time()

    for image_index in range(len(frame.images)):
        t1 = time()

        camera_image = frame.images[image_index]
        image_matrix = imread(camera_image.image)

        # Search for camera[i] in frame.context.camera_calibrations since it does not follow
        # same order as camera_image and camera_labels
        camera_context = next(context for context in frame.context.camera_calibrations
                              if context.name == camera_image.name)
        camera_labels = []
        if frame.camera_labels:
            camera_labels = frame.camera_labels[image_index]

        # Filter projections corresponding to this camera
        mask = np.equal(lidar_projections[:, 1], camera_context.name)
        matching_projections = lidar_projections[mask]

        # Get y, x indices from channels 2,3 and reverse
        indices = np.flip(matching_projections[:, -2:], [1]).astype(int)

        # Get values
        values = matching_projections[:, 0]

        # Normalize according to max values
        values = NORMALIZATION[0](values)

        # Round values and cast to int
        values = np.round(values).astype(np.int32)

        # Create LiDAR channel
        lidar_projection_channel = SKY_VALUE[0] * np.ones(image_matrix.shape[:2])
        lidar_projection_channel[indices[:, 0], indices[:, 1]] = values
        t2 = time()
        if SKY_VALUE[0] == 255:
            lidar_projection_channel = -1 * maximum_filter(-1 * lidar_projection_channel, size=(22, 11), mode="reflect")
        else:
            lidar_projection_channel = maximum_filter(lidar_projection_channel, size=(22, 11), mode="reflect")
        #print("MAX FILTER: ", time() - t2)
        t2 = time()

        if frame_index % frames_to_skip == 0 or is_testing:
            lidar_projection_channel = median_filter(lidar_projection_channel, size=(22, 11), mode="reflect")

        lidar_projection_channel = lidar_projection_channel.astype(np.uint8)
        #print("MEDIAN FILTER: ", time() - t2)

        # lidar_projection_channel_median2 = median_filter(lidar_projection_channel, size=(20, 10), mode="reflect")
        # fig, (ax1, ax2) = plt.subplots(2, 2)
        # ax1[0].imshow(image_matrix)
        # ax1[1].imshow(lidar_projection_channel, cmap='Reds')
        # ax2[0].imshow(lidar_projection_channel_median, cmap='Reds')
        # ax2[1].imshow(lidar_projection_channel_median2, cmap='Reds')
        # plt.show()

        lidar_projection_channel = np.expand_dims(lidar_projection_channel, axis=2)

        additional_channels_encoded = tf.io.encode_jpeg(lidar_projection_channel).numpy()



        tf_example, contains_labels, contains_cyclist = create_tf_example(frame_context=frame.context,
                                       camera_context=camera_context,
                                       camera_image=camera_image,
                                       camera_labels=camera_labels,
                                       timestamp_micros=frame.timestamp_micros,
                                       channels=3,
                                       additional_channels=additional_channels_encoded,
                                       ignore_difficult_instances=ignore_difficult_instances)

        if (frame_index % frames_to_skip == 0 and contains_labels) or contains_cyclist or is_testing:
            tf_examples.append(tf_example)

        #print("1 IMAGE: ", time() - t1)

    # print("-----------------------------------------------")
    # print("5 IMAGES: ", time() - t)
    # print("-----------------------------------------------")

    return tf_examples




def preprocess_rgb_rie(frame, ignore_difficult_instances):
    """ Change red channel for LiDAR projection """
    tf_examples = []


    (range_images, camera_projections, _) = frame_utils.parse_range_image_and_camera_projection(
        frame)

    # Range image
    range_image = range_images[LIDAR][0]
    range_image_numpy = np.asarray(range_image.data)
    range_image_numpy = np.reshape(range_image_numpy, range_image.shape.dims)
    range_image_numpy = range_image_numpy[..., :3]

    # Change shape to (width,height,1) if only one channel
    if len(range_image_numpy.shape) == 2:
        range_image_numpy = np.expand_dims(range_image_numpy, axis=2)

    # Camera projections
    camera_projection = camera_projections[LIDAR][0]
    camera_projection_numpy = np.asarray(camera_projection.data)
    camera_projection_numpy = np.reshape(camera_projection_numpy, camera_projection.shape.dims)
    camera_projection1_numpy = camera_projection_numpy[..., :3]
    camera_projection2_numpy = camera_projection_numpy[..., 3:]


    # Concat lidar and camera projection (Shape: (64,2650,6))
    # Description of 6 channels in lidar_projection1/2:
    # 0: range, 1: intensity, 2: elongation, 3: camera, 4: x position, 5: y position
    lidar_projection1 = np.concatenate([range_image_numpy, camera_projection1_numpy], axis=2)
    lidar_projection2 = np.concatenate([range_image_numpy, camera_projection2_numpy], axis=2)

    # Flatten rows and columns. Resulting shape (169600,6)
    lidar_projection1 = np.reshape(lidar_projection1, [lidar_projection1.shape[0] * lidar_projection1.shape[1], -1])
    lidar_projection2 = np.reshape(lidar_projection2, [lidar_projection2.shape[0] * lidar_projection2.shape[1], -1])

    # Concat first and second projections in one array of shape (339200, 6)
    lidar_projections = np.concatenate([lidar_projection1, lidar_projection2])

    t = time()
    for image_index in range(len(frame.images)):
        t1 = time()

        camera_image = frame.images[image_index]
        image_matrix = imread(camera_image.image)

        # Search for camera[i] in frame.context.camera_calibrations since it does not follow
        # same order as camera_image and camera_labels
        camera_context = next(context for context in frame.context.camera_calibrations
                              if context.name == camera_image.name)

        # Filter projections corresponding to this camera (channel 3 of lidar_projections)
        mask = np.equal(lidar_projections[:, 3], camera_context.name)
        matching_projections = lidar_projections[mask]

        # Get y, x indices from channels 2,3 and reverse
        indices = np.flip(matching_projections[:, -2:], [1]).astype(int)

        # Get values of channels intensity, range, elongation
        values = matching_projections[:, :3]
        # Normalize each channel separately according to NORMALIZATION
        for i in range(3):
            values[:, i] = NORMALIZATION[i](values[:, i])

        # values[:, 0] = 255 * (MAX_LIDAR_MEASURES[0] - values[:, 0])/MAX_LIDAR_MEASURES[0]
        # values[:, 1] = 255 * np.tanh(values[:, 1]),
        # values[:, 2] = 255 * values[:, 2]/MAX_LIDAR_MEASURES[2]

        # Round values and cast to int
        values = np.round(values).astype(np.int32)

        # Create LiDAR channel initialized to ones
        lidar_projection_channels = 0 * np.ones(image_matrix.shape)

        # For each channel: multiply by sky_value and insert values according to indices
        for i in range(3):
            # t = time()
            # TODO: ESTA linea tarda demasiado
            # lidar_projection_channels[..., i] = lidar_projection_channels[..., i] * SKY_VALUE[i]
            # print(time()-t)
            # t = time()
            lidar_projection_channels[indices[:, 0], indices[:, 1], i] = values[:, i]
            # print(time()-t)

        t2 = time()
        lidar_projection_channels = maximum_filter(lidar_projection_channels, size=(22, 11, 1), mode="reflect")
        # print("MAX FILTER: ", time() - t2)

        lidar_projection_channels = lidar_projection_channels.astype(np.uint8)
        # lidar_projection_channel = np.expand_dims(lidar_projection_channel, axis=2)
        additional_channels_encoded = tf.io.encode_jpeg(lidar_projection_channels).numpy()

        camera_labels = frame.camera_labels[image_index]
        tf_example = create_tf_example(frame_context=frame.context,
                                       camera_context=camera_context,
                                       camera_image=camera_image,
                                       camera_labels=camera_labels,
                                       timestamp_micros=frame.timestamp_micros,
                                       channels=3,
                                       additional_channels=additional_channels_encoded,
                                       ignore_difficult_instances=ignore_difficult_instances)
        tf_examples.append(tf_example)
    #     print("1 IMAGE: ", time() - t1)
    # print("-----------------------------------------------")
    # print("5 IMAGES: ", time()-t)
    # print("-----------------------------------------------")
    return tf_examples





PREPROCESSING_METHODS = {
    'rgb': preprocess_rgb,
    'lidarGB': preprocess_lidar_gb,
    'rgbRange': preprocess_rgb_range,
    'rgbRIE': preprocess_rgb_rie,
    None: preprocess_rgb,
}


def get_tf_examples(frame, frame_index, frames_to_skip, ignore_difficult_instances, is_testing, preprocessing=None):
    """
    Extracts and preprocesses the image from a frame
    Args:
        frame: waymo frame
        ignore_difficult_instances: Whether to ignore difficult instances
        preprocessing: how to preprocess the image
            Options:
                'rgb' | None --> returns the RGB image
                'lidarGB'   --> change red chanel with LiDAR projection
            Default: None #return frame.images[image_index]

    Returns:
        preprocessed image
    """
    return PREPROCESSING_METHODS[preprocessing](frame, frame_index, frames_to_skip,
                                                ignore_difficult_instances, is_testing)
