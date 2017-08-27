

import sys
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


#MODEL_NAME = 'tfdroid'

# Freeze the graph

input_graph_path = "/home/lhuynh/Desktop/tf/test1/checkpoint/mnist_model.pb"
input_saver_def_path = ""
input_binary = False
checkpoint_path = "/home/lhuynh/Desktop/tf/test1/checkpoint/mnist_model.ckpt"
output_node_names = "prediction"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_graph_path = "./checkpoint/prediction_frozen.pb"
clear_devices = True

freeze_graph.freeze_graph(
    input_graph_path,
    input_saver_def_path,
    input_binary,
    checkpoint_path,
    output_node_names,
    restore_op_name,
    filename_tensor_name,
    output_graph_path,
    clear_devices, "")

# Optimize for inference
input_graph_def = tf.GraphDef()
with tf.gfile.Open("./checkpoint/prediction_frozen.pb", "r") as f:
    data = f.read()
    input_graph_def.ParseFromString(data)

output_graph_def = optimize_for_inference_lib.optimize_for_inference(
        input_graph_def,
        ["input"], # an array of the input node(s)
        ["prediction"], # an array of output nodes
        tf.float32.as_datatype_enum)

# Save the optimized graph

f = tf.gfile.FastGFile("./checkpoint/prediction_optimize.pb", "w")
f.write(output_graph_def.SerializeToString())

# # tf.train.write_graph(output_graph_def, './', output_optimized_graph_name)
