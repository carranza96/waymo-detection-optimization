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
flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')
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


def add_detections_to_example(raw_example, detections, discard_image_pixels):
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    detection_boxes = detections['detection_boxes'].T

    tf_example = tf.train.Example().FromString(raw_example.numpy())
    feature = tf_example.features.feature
    feature[standard_fields.TfExampleFields.
            detection_score].float_list.value[:] = detections['detection_scores']
    feature[standard_fields.TfExampleFields.
            detection_bbox_ymin].float_list.value[:] = detection_boxes[0]
    feature[standard_fields.TfExampleFields.
            detection_bbox_xmin].float_list.value[:] = detection_boxes[1]
    feature[standard_fields.TfExampleFields.
            detection_bbox_ymax].float_list.value[:] = detection_boxes[2]
    feature[standard_fields.TfExampleFields.
            detection_bbox_xmax].float_list.value[:] = detection_boxes[3]
    feature[standard_fields.TfExampleFields.
            detection_class_label].int64_list.value[:] = detections['detection_classes']

    if discard_image_pixels:
        del feature[standard_fields.TfExampleFields.image_encoded]

    return tf_example


def main(_):

    required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                      'inference_graph']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    if FLAGS.gpu_device:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_device)

    output_folder = "/"
    output_folder = output_folder.join(FLAGS.output_tfrecord_path.split("/")[:-1])

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    input_tfrecord_paths = [
        v for v in FLAGS.input_tfrecord_paths.split(',') if v]
    print('Reading input from %d files', len(input_tfrecord_paths))

    dataset = tf.data.TFRecordDataset(input_tfrecord_paths)
    detect_fn = tf.saved_model.load(FLAGS.inference_graph)

    with tf.io.TFRecordWriter(
            FLAGS.output_tfrecord_path) as tf_record_writer:

        for i, raw_example in enumerate(dataset):
            if i % 10 == 0:
                print('Processed %d images...'%i)
            image_tensor = build_input(raw_example)

            t1 = time()
            detections = detect_fn(image_tensor)
            print(i, time()-t1)

            tf_example = add_detections_to_example(raw_example, detections, FLAGS.discard_image_pixels)

            tf_record_writer.write(tf_example.SerializeToString())

        print('Finished processing records')


if __name__ == '__main__':
    app.run(main)
