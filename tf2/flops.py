import tensorflow as tf
import os
from tensorflow.python.saved_model import tag_constants
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_detection_graph(frozen_graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(frozen_graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.compat.v1.import_graph_def(graph_def, name='')
            # tf.train.import_meta_graph(graph_def)
    return detection_graph



tf.compat.v1.disable_v2_behavior()
model_path = 'saved_models/inference_models/faster_rcnn_resnet50_v1_1280x1920_coco17_tpu-8/saved_model/'
#model_path = 'frozen_inference_graph.pb'
session = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

#tf.compat.v1.reset_default_graph()
with graph.as_default():
    with session.as_default():
        # model = tf.keras.models.load_model(model_path)
        # model = tf.compat.v1.keras.experimental.load_from_saved_model(model_path)
        model = tf.saved_model.load(model_path)
        # model = tf.compat.v1.saved_model.load(export_dir=model_path, sess=session, tags=[tf.compat.v1.python.saved_model.tag_constants.SERVING])
        #graph = load_detection_graph(model_path)

        # saver_kwargs = {}
        # saver = tf.train.Saver(**saver_kwargs)
        # input_saver_def = saver.as_saver_def()

        # write_graph_and_checkpoint(
        #     inference_graph_def=tf.get_default_graph().as_graph_def(),
        #     model_path=model_path,
        #     input_saver_def=input_saver_def,
        #     trained_checkpoint_prefix=checkpoint_to_use)

        inference_graph_def = tf.compat.v1.get_default_graph().as_graph_def()
        inference_graph_path = os.path.join('',
                                            'inference_graph.pbtxt')
        for node in inference_graph_def.node:
            node.device = ''
        with tf.compat.v1.gfile.GFile(inference_graph_path, 'wb') as f:
            f.write(str(inference_graph_def))

        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

        # We use the Keras session graph in the call to the profiler.
        flops = tf.compat.v1.profiler.profile(graph=graph,
                                              run_meta=run_meta, cmd='op', options=opts)

print(flops.total_float_ops)


# full_model = detect_fn.get_concrete_function(image=tf.TensorSpec(input_tensor.shape, input_tensor.dtype))
# frozen_func = convert_variables_to_constants_v2(full_model)
# # frozen_func.graph.as_graph_def()
# # layers = [op.name for op in frozen_func.graph.get_operations()]
# flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph, run_meta=metadata, cmd='op', options=opts3)
#
# # tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
# #                   logdir="./frozen_models",
# #                   name="prueba.pb",
# #                   as_text=False)


# opts = tf.compat.v1.profiler.ProfileOptionBuilder.time_and_memory()
# opts2 = tf.compat.v1.profiler.ProfileOptionBuilder.trainable_variables_parameter()
# opts3 = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()



# def profile_inference_graph(graph):
#   """Profiles the inference graph.
#
#   Prints model parameters and computation FLOPs given an inference graph.
#   BatchNorms are excluded from the parameter count due to the fact that
#   BatchNorms are usually folded. BatchNorm, Initializer, Regularizer
#   and BiasAdd are not considered in FLOP count.
#
#   Args:
#     graph: the inference graph.
#   """
#   tfprof_vars_option = (
#       contrib_tfprof.model_analyzer.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
#   tfprof_flops_option = contrib_tfprof.model_analyzer.FLOAT_OPS_OPTIONS
#
#   # Batchnorm is usually folded during inference.
#   tfprof_vars_option['trim_name_regexes'] = ['.*BatchNorm.*']
#   # Initializer and Regularizer are only used in training.
#   tfprof_flops_option['trim_name_regexes'] = [
#       '.*BatchNorm.*', '.*Initializer.*', '.*Regularizer.*', '.*BiasAdd.*'
#   ]
#
#   contrib_tfprof.model_analyzer.print_model_analysis(
#       graph, tfprof_options=tfprof_vars_option)
#
#   contrib_tfprof.model_analyzer.print_model_analysis(
#       graph, tfprof_options=tfprof_flops_option)