import tensorflow as tf
import os
import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
import gc
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.core import standard_fields
from tensorflow.python.eager import context
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


## IMPORTANT: Requires TF 2.0
## TODO: Read img from tf record

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def get_model_detection_function(model):
    """Get a tf.function for detection."""

    @tf.function
    def detect_fn(image):
        """Detect objects in image."""

        image, shapes = model.preprocess(image)
        prediction_dict = model.predict(image, shapes)
        detections = model.postprocess(prediction_dict, shapes)

        return detections, prediction_dict, tf.reshape(shapes, [-1])

    return detect_fn


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

def get_stats(model_name):
    pipeline_config = 'saved_models/inference_models/' + model_name + '/pipeline.config'
    model_dir = 'saved_models/inference_models/' + model_name + '/checkpoint/'

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    detect_fn = get_model_detection_function(detection_model)
    image_np = load_image_into_numpy_array("image.png")
    input_tensor = tf.convert_to_tensor(
        np.expand_dims(image_np, 0), dtype=tf.float32)

    # Number of parameters
    variables = tf.train.list_variables(model_dir)
    total_parameters = 0
    for variable in variables:
        # shape is an array of tf.Dimension
        shape = variable[1]
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    # Memory usage
    with context.eager_mode():
        context.enable_run_metadata()
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()
        profiler = tf.compat.v1.profiler.Profiler()
        metadata = context.export_run_metadata()
        profiler.add_step(0, metadata)
        context.disable_run_metadata()
        tm = profiler.profile_graph(opts)
        memory = tm.total_requested_bytes

    # Number of flops
    full_model = detect_fn.get_concrete_function(image=tf.TensorSpec(input_tensor.shape, input_tensor.dtype))
    frozen_func = convert_variables_to_constants_v2(full_model)
    # frozen_func.graph.as_graph_def()
    # layers = [op.name for op in frozen_func.graph.get_operations()]
    stats = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=metadata, cmd='op',
                                          options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
    flops = stats.total_float_ops

    stats = {'model_name': model_name, 'parameters': total_parameters, 'flops': flops, 'memory': memory}
    return stats



df = pd.DataFrame(columns=["model_name", "parameters", "flops", "memory"])
dataset = tf.data.TFRecordDataset("data/camera_data/training/training.record-00000-of-00798")

for i, model_name in enumerate(os.listdir('saved_models/inference_models/')):
    row = get_stats(model_name)
    df = df.append(row, ignore_index=True)

df.to_csv('profile.csv')
