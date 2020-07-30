import os
import numpy as np
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

## TODO: Align these features with dataset creation
def parse_camera_tfrecord_example(data, additional_channels=False, with_detections=False):
    features = {
        "image/height": tf.io.FixedLenFeature((), tf.int64, default_value=1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, default_value=1),
        "image/encoded": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/format": tf.io.FixedLenFeature((), tf.string, default_value='jpeg'),
        "image/source_id": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/context_name": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/frame_timestamp_micros": tf.io.FixedLenFeature((), tf.int64, default_value=1),
        "image/camera_name": tf.io.FixedLenFeature((), tf.int64, default_value=1),
        "image/time_of_day": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/location": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/weather": tf.io.FixedLenFeature((), tf.string, default_value=''),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/text": tf.io.VarLenFeature(tf.string),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/difficult": tf.io.VarLenFeature(tf.int64),
    }

    if with_detections:
        features.update({
            "image/detection/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/detection/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/detection/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/detection/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/detection/score": tf.io.VarLenFeature(tf.float32),
            "image/detection/label": tf.io.VarLenFeature(tf.int64),
        })

    if additional_channels:
        features.update({
            "image/additional_channels/encoded": tf.io.FixedLenFeature((), tf.string, default_value='')
        })
    # decode the TFRecord
    tf_record = tf.io.parse_single_example(data, features)

    # VarLenFeature fields require additional sparse.to_dense decoding
    var_len_features = [feature_name for feature_name, feature in features.items()
                        if isinstance(feature, tf.io.VarLenFeature)]
    for feature_name in var_len_features:
        default_value = '' if tf_record[feature_name].dtype == tf.string else 0
        tf_record[feature_name] = tf.sparse.to_dense(tf_record[feature_name], default_value)

    return tf_record


def read_and_parse_sharded_dataset(filename_pattern, additional_channels=False):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = True

    dataset = tf.data.Dataset.list_files(filename_pattern, seed=20)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=32,
                                 num_parallel_calls=tf.data.experimental.AUTOTUNE)
    print(additional_channels)
    return dataset.map(lambda data: parse_camera_tfrecord_example(data, additional_channels))


def get_dataset_class_distribution(dataset):

    def label_count(counter, example):

        labels = example["image/object/class/label"]

        def tf_count(tensor, val):
            elements_equal_to_value = tf.equal(tensor, val)
            as_ints = tf.cast(elements_equal_to_value, tf.int32)
            return tf.reduce_sum(as_ints)

        counter['TYPE_VEHICLE'] += tf_count(labels, 1)
        counter['TYPE_PEDESTRIAN'] += tf_count(labels, 2)
        counter['TYPE_CYCLIST'] += tf_count(labels, 3)

        return counter

    initial_counter = {'TYPE_VEHICLE': 0, 'TYPE_PEDESTRIAN': 0, 'TYPE_CYCLIST': 0}
    class_distribution = dataset.reduce(initial_state=initial_counter, reduce_func=label_count)

    # Get fractions
    sum_elements = int(tf.add_n(list(class_distribution.values())))
    for k, v in class_distribution.items():
        v = v.numpy()
        class_distribution[k] = (v, v/sum_elements)

    return class_distribution


def read_frame_waymo_segment(tf_record_path, frame_index=0):
    dataset = tf.data.TFRecordDataset(tf_record_path, compression_type='')
    frame = None
    for i, data in enumerate(dataset):
        if i == frame_index:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
    return frame


def write_serialized_string(file, objects):
    f = open(file, 'wb')
    serialized = objects.SerializeToString()
    f.write(serialized)

