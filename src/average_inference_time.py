import tensorflow as tf
from utils_tf_record.read_dataset_utils import read_and_parse_sharded_dataset, parse_camera_tfrecord_example
from absl import app
from absl import flags
from time import time
import numpy as np

tf.enable_eager_execution()

# FLAGS = flags.FLAGS
# flags.DEFINE_string('inference_graph_path', 'saved_models/final_models/inference_models/default_anchors_500_256/frozen_inference_graph.pb',
#                     'Path to the inference graph')
# flags.DEFINE_string('dataset_file_pattern', "data/camera_lateral/training/*",
#                     'TFRecord file containing ground truths and detections')
# flags.DEFINE_string('metrics_file', 'data/time.csv', "Metrics csv file to write average inference time")
# flags.DEFINE_integer('num_images', 1000, "Number of images to test")
# tf.flags.DEFINE_integer('num_additional_channels', 0, 'Number of additional channels to use')


# Load the Tensorflow model into memory.
def load_detection_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
            # tf.train.import_meta_graph(graph_def)
    return detection_graph


def read_encoded_image(data, num_additional_channels):
    image = tf.image.decode_jpeg(data['image/encoded'])
    if num_additional_channels > 0:
        additional_channels = tf.image.decode_jpeg(data['image/additional_channels/encoded'])
        image = tf.concat([image, additional_channels], axis=2)
    return image.numpy()


def run_inference_for_single_image(sess, detection_graph, image):
    # Define input and output tensors (i.e. data) for the object detection classifier
    # Input tensor is the image
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # The model expects a batch of images, so add an axis
    image_expanded = np.expand_dims(image, axis=0)

    # Output tensors are the detection boxes, scores, and classes
    # Each box represents a part of the image where a particular object was detected
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represents level of confidence for each of the objects.
    # The score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Number of objects detected
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')


    # Run inference
    inference_times = []
    for i in range(3):
        t1 = time()
        (boxes, scores, classes, n_detections) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_expanded})
        t2 = time()
        inference_times.append(t2-t1)
        # print(t2-t1)

    # print("Average inference time (ms) :", np.mean(inference_times[1:]))
    avg_inf_time = np.mean(inference_times[1:])
    num_detections = int(n_detections)

    # Take out batch dimension and get first num_detections
    boxes = np.squeeze(boxes)[:num_detections]
    scores = np.squeeze(scores)[:num_detections]
    classes = np.squeeze(classes)[:num_detections].astype(np.int32)

    return boxes, scores, classes, num_detections, avg_inf_time


# def main(_):

detection_graph = load_detection_graph('saved_models/final_models/inference_models/default_anchors_500_256/frozen_inference_graph.pb')
# detection_graph = load_detection_graph('saved_models/final_models/inference_models/measure_time/frozen_inference_graph.pb')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(graph=detection_graph, config=config)

# dataset = read_and_parse_sharded_dataset(FLAGS.dataset_file_pattern,
#                                          additional_channels=bool(FLAGS.num_additional_channels))

# filename = "data/camera_lateral/training/lateral_training.record-00000-of-00798"
filename = "data/camera_lateral/training/lateral_training.record-00000-of-00798"
dataset = tf.data.TFRecordDataset(filename, compression_type='')
dataset = dataset.map(lambda data: parse_camera_tfrecord_example(data, False))


inference_times = []
for i, d in enumerate(dataset):
    image = read_encoded_image(d, 0)

    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num, avg_inf_time) = run_inference_for_single_image(sess, detection_graph, image)
    inference_times.append(avg_inf_time)
    if i == 1000:
        break

    avg_time = np.mean(inference_times[1:])

print("AVERAGE INFERENCE TIME:%.6f" % avg_time)
# f = open(FLAGS.metrics_file, 'a')
# f.write("INFERENCE TIME,%.6f" % avg_time)


# if __name__ == '__main__':
#     app.run(main)
