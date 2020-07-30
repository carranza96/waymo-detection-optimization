r"""Convert raw Waymo Open dataset 2D camera data to TFRecord for object_detection.

Example usage:
    python create_camera_waymo_tf_record.py \
        --set=training_validation \
        --preprocessing=rgb \
        --num_cores=8 \
        --data_dir=data/raw_data/ \
        --output_path=data/camera_data/

Data_dir structure:
    data/
        raw_data/
            training/
                training_0000/
                    segment-272435602399417322_2884_130_2904_130_with_camera_labels.tfrecord
                    segment-662188686397364823_3248_800_3268_800_with_camera_labels.tfrecord
                    ...
                training_0001/
                    segment-902001779062034993_2880_000_2900_000_with_camera_labels.tfrecord
                    ...
                ...
            validation/
                validation_0000/
                    segment-1105338229944737854_1280_000_1300_000_with_camera_labels.tfrecord
                ...

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v1 as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from src.utils.waymo_frame_preprocessing import get_tf_examples, PREPROCESSING_METHODS
import contextlib2
from tqdm import tqdm
from multiprocessing import Pool
from time import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.enable_eager_execution()


flags = tf.app.flags
flags.DEFINE_string('set', 'training_validation', 'Convert training set, validation set or both')
flags.DEFINE_string('preprocessing', 'rgbRange', 'Preprocessing method the image')
flags.DEFINE_string('data_dir', 'data/raw_data/', 'Root directory to raw Waymo Open dataset.')
flags.DEFINE_string('output_path', 'data/camera_data/', 'Path to output TFRecord')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore difficult instances')
flags.DEFINE_integer('num_cores', 8, 'Number of CPU cores to use for multiprocessing execution')
flags.DEFINE_integer('frames_to_skip', 0, 'Skip every n frames')

FLAGS = flags.FLAGS

# TODO: Add test set
# TODO: Split training to allow multiple processes
SETS = ['training', 'validation', 'training_validation', 'testing']


def open_sharded_output_tfrecords(exit_stack, base_path, num_shards):
    """Opens all TFRecord shards for writing and adds them to an exit stack.

    Args:
      exit_stack: A context2.ExitStack used to automatically closed the TFRecords
        opened in this function.
      base_path: The base path for all shards
      num_shards: The number of shards

    Returns:
      The list of opened TFRecords. Position k in the list corresponds to shard k.
    """
    tf_record_output_filenames = [
        '{}-{:05d}-of-{:05d}'.format(base_path, idx, num_shards)
        for idx in range(num_shards)
    ]

    # options = tf.io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)

    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(file_name))
        for file_name in tf_record_output_filenames
    ]

    return tfrecords


def _create_frame(raw_frame_data):
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(raw_frame_data.numpy()))
    return frame

def main(_):
    # Check flag preprocessing method
    if FLAGS.preprocessing not in PREPROCESSING_METHODS:
        raise ValueError('preprocessing must be in : {}'.format(PREPROCESSING_METHODS))
    preprocessing_method = FLAGS.preprocessing

    # Check flag training/validation set
    if FLAGS.set not in SETS:
        raise ValueError('set must be in : {}'.format(SETS))

    sets = [FLAGS.set]
    # if FLAGS.set == 'training_validation':
    #     sets = ['training', 'validation']

    for set_type in sets:
        # Append training/validation to data and output path
        data_dir = FLAGS.data_dir + set_type
        output_path = FLAGS.output_path + set_type
        # Get all paths to .tfrecords files
        files_list = []
        for (dirpath, dirnames, filenames) in os.walk(data_dir):
            files_list += [os.path.join(dirpath, file) for file in filenames if "camera_labels.tfrecord" in file]

        # Create output directory if it does not exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        frames_to_skip = FLAGS.frames_to_skip
        is_testing = FLAGS.set not in ["validation", "training"]
        # Shard dataset to multiple output files
        num_shards = len(files_list)
        output_filebase = output_path + "/lateral_" + set_type + ".record"
        counter = 0
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = open_sharded_output_tfrecords(
                tf_record_close_stack, output_filebase, num_shards)

            for shard_index, file in enumerate(tqdm(files_list)):
                segment_dataset = tf.data.TFRecordDataset(file)

                t = time()
                frames = [_create_frame(raw_frame_data) for i, raw_frame_data in enumerate(segment_dataset)]
                print("Create Frames:", time()-t)

                print(len(frames))

                if FLAGS.num_cores > 1:
                    with Pool(processes=FLAGS.num_cores) as pool:
                        tf_examples_ls = pool.starmap(get_tf_examples, [(frame, index, frames_to_skip, False, is_testing
                                                                         , preprocessing_method) for index, frame
                                                                        in enumerate(frames)])
                else:
                    tf_examples_ls = [get_tf_examples(frame, index, frames_to_skip, ignore_difficult_instances=False,
                                                      is_testing=is_testing, preprocessing=preprocessing_method)
                                                      for index, frame in enumerate(frames)]

                print([len(tf_examples) for tf_examples in tf_examples_ls])
                print(sum([1 if len(tf_examples) > 0 else 0 for tf_examples in tf_examples_ls]))
                print(sum([len(tf_examples) for tf_examples in tf_examples_ls]))

                for tf_examples in tf_examples_ls:
                    counter += len(tf_examples)
                    for tf_example in tf_examples:
                        output_tfrecords[shard_index % num_shards].write(tf_example.SerializeToString())

                del frames, tf_examples_ls
    print(counter)


if __name__ == '__main__':
    tf.app.run()






