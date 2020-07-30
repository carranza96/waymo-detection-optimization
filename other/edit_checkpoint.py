from tensorflow.python import pywrap_tensorflow
import numpy as np
import tensorflow as tf
import os
# flags = tf.app.flags
# flags.DEFINE_string('input_path',
#                     '/data/object_detection-pretrained-ckpt/faster_rcnn_resnet101_coco_11_06_2017/model.ckpt',
#                     'path of pretrained_checkpoint')
# flags.DEFINE_string('output_path', 'ckpts/model.ckpt', 'output checkpoint')
# flags.DEFINE_string('feature_extractor', 'resnet_v1_101', 'name of first checkpoint')


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# INPUT_PATH = "saved_models/final_models/best_high_res/model.ckpt-2285803"
# INPUT_PATH = "saved_models/sample/ao_frcnn_500_256_extra_features/model.ckpt-200000"
INPUT_PATH = "saved_models/final_models/aofinal_frcnn_500_256_extrafeat_redsoftmax_bikes_higherlr/model.ckpt-200000"

OUTPUT_PATH = "saved_models/final_models/redsoftmax_bikes_modified_rgbR/model.ckpt"

reader = pywrap_tensorflow.NewCheckpointReader(INPUT_PATH)
var_to_shape_map = reader.get_variable_to_shape_map()

new_vars = []


var_to_edit_names = ['FirstStageFeatureExtractor/resnet_v1_101/conv1/weights']

for key in sorted(var_to_shape_map):
    if key not in var_to_edit_names:
        var = tf.Variable(reader.get_tensor(key), name=key, dtype=tf.float32)
    else:
        print("Found variable: {}".format(key))
vars_to_edit = []
for name in var_to_edit_names:
    if reader.has_tensor(name):
        vars_to_edit.append(reader.get_tensor(name))
    else:
        raise Exception("{} not found in checkpoint. Check feature extractor name. Exiting.".format(name))
for name, var_to_edit in zip(var_to_edit_names, vars_to_edit):
    # np.save("saved_models/final_models/best_high_res_modified_checkpoint/" + name.replace("/", "_") + "_extra_features.npy", var_to_edit[2048:])
    mean_3channels = np.expand_dims(np.mean(var_to_edit, axis=2), axis=2)
    new_var = np.concatenate((var_to_edit, mean_3channels), axis=2)
    new_vars.append(tf.Variable(new_var, name=name, dtype=tf.float32))



# ## Extra features
# var_to_edit_names = ['SecondStageBoxPredictor/BoxEncodingPredictor/weights',
#                      'SecondStageBoxPredictor/ClassPredictor/weights']
#
# for key in sorted(var_to_shape_map):
#     if key not in var_to_edit_names:
#         var = tf.Variable(reader.get_tensor(key), name=key, dtype=tf.float32)
#     else:
#         print("Found variable: {}".format(key))
# vars_to_edit = []
# for name in var_to_edit_names:
#     if reader.has_tensor(name):
#         vars_to_edit.append(reader.get_tensor(name))
#     else:
#         raise Exception("{} not found in checkpoint. Check feature extractor name. Exiting.".format(name))
# for name, var_to_edit in zip(var_to_edit_names, vars_to_edit):
#     # np.save("saved_models/final_models/best_high_res_modified_checkpoint/" + name.replace("/", "_") + "_extra_features.npy", var_to_edit[2048:])
#     extra_feat_weights = np.load("saved_models/final_models/best_high_res_modified_checkpoint/" + name.replace("/", "_") + "_extra_features.npy")
#     new_var = np.concatenate([var_to_edit, extra_feat_weights])
#     new_vars.append(tf.Variable(new_var, name=name, dtype=tf.float32))



# var_to_edit_names =['FirstStageBoxPredictor/BoxEncodingPredictor/biases',
#                     'FirstStageBoxPredictor/BoxEncodingPredictor/weights',
#                     'FirstStageBoxPredictor/ClassPredictor/biases',
#                     'FirstStageBoxPredictor/ClassPredictor/weights']
#
# print('Loading checkpoint...')
# for key in sorted(var_to_shape_map):
#     if key not in var_to_edit_names:
#         var = tf.Variable(reader.get_tensor(key), name=key, dtype=tf.float32)
#     else:
#         print("Found variable: {}".format(key))
# vars_to_edit = []
# for name in var_to_edit_names:
#     if reader.has_tensor(name):
#         vars_to_edit.append(reader.get_tensor(name))
#     else:
#         raise Exception("{} not found in checkpoint. Check feature extractor name. Exiting.".format(name))
#
#
#
# for name, var_to_edit in zip(var_to_edit_names, vars_to_edit):
#     for i in range(5):
#         new_name = name.split("/")
#         new_name.insert(1, "BoxPredictor_" + str(i))
#         new_name = '/'.join(new_name)
#         new_vars.append(tf.Variable(var_to_edit, name=new_name, dtype=tf.float32))


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, OUTPUT_PATH)

# Only need .0000-of-0001 and .index file. Good to go!
