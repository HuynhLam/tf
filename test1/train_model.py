'''
Lam Huynh, Oulu 2017
Tackle MNIST problem
1. Train a simple logistic regression model
2. Save its weights and meta-graph after trained
3. Load the trained weights and meta-graph, then use to calculate the test result
(Using hyperparams file for trainings)
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data

import hyper_params as my_params

def main():
    # Load the MNIST data
    mnist = input_data.read_data_sets("mnist", one_hot=True)

    # Define the model
    X = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="input")
    input_layer = tf.reshape(X, [-1, 28, 28, 1], name="input_layer")

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5,5], padding="same", activation=tf.nn.relu, name="conv_layer_1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5,5], padding="same", activation=tf.nn.relu, name="conv_layer_2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)

    conv1_flatten = tf.reshape(pool2, shape=[-1, 7 * 7 * 64], name="flatten_before_dense")
    dense = tf.layers.dense(inputs=conv1_flatten, units=1024, activation=tf.nn.relu, name="first_dense")
    #dense1 = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.relu, name="second_dense")
    dropout1 = tf.layers.dropout(inputs=dense, rate=my_params.drop_rate, name="dropout1")

    Y = tf.layers.dense(inputs=dropout1, units=10, name="last_dense")
    print("Yes, finished calculate the last dense!!")

    Y_labels = tf.placeholder(tf.float32, shape=[None, 10], name="y_labels")

    # Define loss function and optimizer
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_labels))
    opt = tf.train.GradientDescentOptimizer(my_params.lr)
    train_step = opt.minimize(xent)

    # Define session and initialize variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training
    losts = []
    # Using plt.ion to interactively plot the chart to visualize how the loss doing
    # Useful when doing in small iteration ~1000
    plt.ion()
    for i in range(my_params.steps+1):
        [batch_x, batch_y] = mnist.train.next_batch(100)
        [loss, _] = sess.run([xent, train_step], feed_dict={X: batch_x, Y_labels: batch_y})
        print("iter: {0}, l = {1}".format(i, loss))
        losts.append(loss)
        plt.gca().cla() # optionally clear axes
        plt.plot(np.linspace(0, i, i+1), losts, label="xent")
        plt.legend()
        plt.draw()
        plt.pause(0.001)

    # Save the meta-graph and trained weights
    # Check if did the folder exist ? If not, create it
    # os.path.isdir return True if path is an existing directory
    # if not os.path.isdir(my_params.checkpoint_path):
    #     os.mkdir(my_params.checkpoint_path)
    # saver = tf.train.Saver()
    # save_path = my_params.checkpoint_path + '/mnist_lr=' + str(my_params.lr)
    # print("save path : ", save_path)
    # saver.save(sess, save_path, global_step=my_params.steps)

    # Testing
    print("********* This is testing !!!")
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_labels: mnist.test.labels}))
    print("********* Finished testing !!!")

    # Plot the loss chart and see how it goes
    plt.plot(np.linspace(0, my_params.steps, my_params.steps+1), losts, label="final_xent")
    plt.legend()
    plt.show(block=True)

if(__name__ == "__main__"):
    main()
