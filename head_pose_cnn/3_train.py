'''
Lam Huynh
Cifar 10 CNN - 3 layers
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
#import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

os.environ['CUDA_VISIBLE_DEVICES'] = "6,7"

def load_data():
    data = input_data.read_data_sets("mnist", one_hot=True)
    return data

def creat_cnn_model(features, labels, mode):
    ''' Build a simple CNN model with 2 cnn 2 pool 2 FC layers '''

    # Input layers
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Conv layer #1
    conv1 = tf.layers.conv2d(inputs=input_layer,
                             filters=32,
                             kernel_size=[5,5],
                             padding="SAME",
                             activation=tf.nn.relu)

    # Pooling layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2,2],
                                    strides=2)

    # Layer #2
    conv2 = tf.layers.conv2d(inputs=pool1,
                            filters=64,
                            kernel_size=[5,5],
                            padding="SAME",
                            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2,2],
                                    strides=2)

    # Dense layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense,
                                rate=0.4,
                                training=mode == learn.ModeKeys.TRAIN)

    # Logit layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
          loss=loss,
          global_step=tf.contrib.framework.get_global_step(),
          learning_rate=0.001,
          optimizer="SGD")

    # Generate Predictions
    predictions = {
        "classes": tf.argmax(
            input=logits, axis=1),
        "probabilities": tf.nn.softmax(
            logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
        mode=mode, predictions=predictions, loss=loss, train_op=train_op)

def main():
    # Load data
    # mnist = load_data()
    # Load training and eval data
    mnist = learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # # Create model
    # X = tf.placeholder(tf.float32, shape=(None, 784), name = "X")
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    # Y_label = tf.placeholder(tf.float32, shape=(None, 10), name = "Y_label")
    #
    # Y = tf.matmul(X, W) + b

    # Create the Estimator
    mnist_classifier = learn.Estimator(model_fn=creat_cnn_model, model_dir="/tmp/mnist_convnet_model")

    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    mnist_classifier.fit(x=train_data,
                        y=train_labels,
                        batch_size=100,
                        steps=1000,
                        monitors=[logging_hook])

    # Configure the accuracy metric for evaluation
    metrics = {
        "accuracy":
            learn.MetricSpec(
                metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }
    # Evaluate the model and print results
    eval_results = mnist_classifier.evaluate(
        x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)

    # # Define optimizer and loss
    # xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_label))
    # opt = tf.train.GradientDescentOptimizer(0.5, name="GradientDescent")
    # train_step = opt.minimize(xent)
    #
    # # Define session and init variables
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    #
    # # Traing
    # losts = []
    # for i in range(5001):
    #     [batch_x, batch_y] = mnist.train.next_batch(100)
    #     [loss, _] = sess.run([xent, train_step], feed_dict={X: batch_x, Y_label: batch_y})
    #     if i % 10 == 0:
    #         print("Iteration: {0}, loss = {1}".format(i, loss))
    #         losts.append(loss)
    #
    # # Plot the loss
    # plt.plot(np.linspace(0, 5000, 501), losts, label="xent")
    # plt.legend()
    # plt.show()

    # # Testing
    # correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_label, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # print("Accuracy = {0}", sess.run(accuracy, feed_dict={X: mnist.test.images, Y_label: mnist.test.labels}))


if __name__ == "__main__":
    main()
