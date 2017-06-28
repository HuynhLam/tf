'''
Using to restore a test MNIST model
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

PATH = "saved_model/"

def main():
    # Load data
    mnist_data = input_data.read_data_sets("mnist", one_hot="True")

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    Y = tf.matmul(x, W) + b
    Y_ = tf.placeholder(tf.float32, [None, 10]) # labels

    sess = tf.Session()
    saver = tf.train.Saver()
    #saver = tf.train.import_meta_graph("m1/my_mnist.ckpt.meta")
    saver.restore(sess, PATH + "my_mnist.ckpt")

    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(acc, feed_dict={x: mnist_data.test.images, Y_: mnist_data.test.labels}))

    return

if(__name__=="__main__"):
    main()
