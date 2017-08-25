'''
Train
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import time

import tensorflow as tf

from timer import Timer
from tensorflow.examples.tutorials.mnist import input_data

# Setup GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

def main():
    # Load the data
    mnist_data = input_data.read_data_sets("mnist", one_hot="True")

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    Y = tf.matmul(x, W) + b
    Y_ = tf.placeholder(tf.float32, [None, 10])

    # Define loss and optimizer
    xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_))
    opt = tf.train.AdamOptimizer(0.001)
    train_step = opt.minimize(xent)

    # Define Session and init variables
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # Training
    with Timer() as t:
        for i in range(20001):
            [batch_x, batch_y] = mnist_data.train.next_batch(100)
            [loss, _]=sess.run([xent, train_step], feed_dict={x: batch_x, Y_: batch_y})
    print("=> Training eslaped: {0:.2f} sec".format(t.secs))

    # Evaluation
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(acc, feed_dict={x: mnist_data.test.images, Y_: mnist_data.test.labels}))

    return

if(__name__=="__main__"):
    main()
