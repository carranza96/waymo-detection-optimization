# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import itertools
import tensorflow.compat.v2 as tf
from object_detection.inference import detection_inference
import os
from absl import flags
from absl import app
from object_detection.core import standard_fields
import numpy as np
from time import time


# tf.enable_v2_behavior()

flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
flags.DEFINE_integer('num_images', 1000, "Number of images to test")
flags.DEFINE_string('metrics_file', None, "Metrics csv file to write average inference time")
flags.DEFINE_integer('gpu_device', None, 'Select GPU device')
flags.DEFINE_integer('num_additional_channels', 0, 'Number of additional channels to use')

FLAGS = flags.FLAGS


def build_input(raw_example):
    features = tf.io.parse_single_example(
        raw_example,
        features={
            standard_fields.TfExampleFields.image_encoded:
                tf.io.FixedLenFeature([], tf.string),
        })
    encoded_image = features[standard_fields.TfExampleFields.image_encoded]
    image_tensor = tf.image.decode_image(encoded_image, channels=3)
    image_tensor.set_shape([None, None, 3])
    image_tensor = tf.expand_dims(image_tensor, 0)

    return image_tensor


def main(_):

    required_flags = ['input_tfrecord_paths',
                      'inference_graph']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    if FLAGS.gpu_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_device)

    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    print('Reading input from %d files', len(input_tfrecord_paths))

    dataset = tf.data.TFRecordDataset(input_tfrecord_paths)
    detect_fn = tf.saved_model.load(FLAGS.inference_graph)

    inference_times = []
    for i, raw_example in enumerate(dataset):

        image_tensor = build_input(raw_example)

        t1 = time()
        _ = detect_fn(image_tensor)
        t2 = time()
        inference_times.append(t2-t1)

        if i == FLAGS.num_images:
            break

    avg_time = np.mean(inference_times[10:])

    print("AVERAGE INFERENCE TIME:%.6f" % avg_time)
    if FLAGS.metrics_file:
        f = open(FLAGS.metrics_file, 'a')
        f.write("INFERENCE TIME,%.6f" % avg_time)


if __name__ == '__main__':
    app.run(main)
