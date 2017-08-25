'''
Lam Huynh, Oulu 2017

Try to do following tasks:
(1. Train a simple logistic regression model)
(2. Save its weights and meta-graph after trained)
3. Load the trained weights and meta-graph, then use to calculate the test result
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from timer import Timer

import hyper_params as my_params

def main():
    # Load MNIST data
    mnist = input_data.read_data_sets("mnist", one_hot=True)

    # Import the meta-graph and its trained weights
    with tf.Session() as sess:
        restore_saver = tf.train.import_meta_graph(my_params.checkpoint_path + '/mnist_lr=' + str(my_params.lr) + "-" + str(my_params.steps) + ".meta")
        # There are two ways to restore here
        # 1. Using tf.train.latest_checkpoint to map with the newest saved checkpoint
        # 2. Using tf.train.Saver.restore to load a specific checkpoint
        #restore_saver.restore(sess, tf.train.latest_checkpoint('./'))
        restore_saver.restore(sess, my_params.checkpoint_path + '/mnist_lr=' + str(my_params.lr) + "-" + str(my_params.steps))
        print("******* Successfully import the trained model {0} YEAH !! ".format(my_params.checkpoint_path + "/mnist_lr=" + str(my_params.lr) + "-" + str(my_params.steps)) )

        # Then load what we need
        graph = tf.get_default_graph()
        # get_tensor_by_name apply for tensor/placeholder/Variables
        # for ops we have get_operation_by_name
        X = graph.get_tensor_by_name("input:0") # placeholder which contain inputs
        input_layer = graph.get_tensor_by_name("input_layer:0")
        # conv1 = graph.get_tensor_by_name("conv_layer_1:0")
        # pool1 = graph.get_tensor_by_name("max_pooling_1:0")
        # conv2 = graph.get_tensor_by_name("conv_layer_2:0")
        # pool2 = graph.get_tensor_by_name("max_pooling_2:0")
        # conv1_flatten = graph.get_tensor_by_name("flatten_before_dense:0")
        # dense = graph.get_tensor_by_name("first_dense:0")
        # dropout1 = graph.get_tensor_by_name("dropout1:0")
        # Y = graph.get_tensor_by_name("last_dense:0")
        Y_labels = graph.get_tensor_by_name("Y_labels:0") # placeholder which contain labels
        #correct_prediction = graph.get_tensor_by_name("correct_prediction:0")
        accuracy = graph.get_tensor_by_name("accuracy:0")

        # Now we perform prediction using the imported model
        # correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_labels, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        for i in range(100):
            [batch_x, batch_y] = mnist.test.next_batch(1000)
            print(sess.run(accuracy, feed_dict={X: batch_x, Y_labels: batch_y}))

if(__name__ == "__main__"):
    with Timer() as tm:
        main()
    print("=> eslaped time: {0:.2f} sec".format(tm.secs))
