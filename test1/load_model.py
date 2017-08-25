'''
Lam Huynh, Oulu 2017
Tackle MNIST problem
1. Train a simple logistic regression model
2. Save its weights and meta-graph after trained
3. Load the trained weights and meta-graph, then use to calculate the test result
This script try to load and use the trained weights and saved meta-graph
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

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
        print("Successfully import the trained model {0}. YEAH !! ".format(my_params.checkpoint_path + "/mnist_lr=" + str(my_params.lr) + "-" + str(my_params.steps)) )

        # Then load what we need
        graph = tf.get_default_graph()
        # get_tensor_by_name apply for tensor/placeholder/Variables
        # for ops we have get_operation_by_name
        X = graph.get_tensor_by_name("input:0") # placeholder which contain inputs
        W = graph.get_tensor_by_name("weights:0") # Variable
        b = graph.get_tensor_by_name("bias:0") # Variable
        Y = graph.get_tensor_by_name("Y_matmul:0") # Variable
        Y = Y + b
        Y_labels = graph.get_tensor_by_name("y_labels:0") # placeholder which contain labels

        # Now we perform prediction using the imported model
        correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
        print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_labels: mnist.test.labels}))

if(__name__ == "__main__"):
    main()
