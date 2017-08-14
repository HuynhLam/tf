'''
Using to save the trained model to disk
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

PATH = "saved_model/"

def main():
    # Load the data
    mnist_data = input_data.read_data_sets("mnist", one_hot="True")

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    Y = tf.matmul(x, W) + b
    Y_ = tf.placeholder(tf.float32, [None, 10]) # labels

    # xent & optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y, labels=Y_))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # init variables & seesion & saver
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    # Training
    for i in range(2001):
        [batch_x, batch_y] = mnist_data.train.next_batch(100)
        [t_loss, _] = sess.run([loss, train_step], feed_dict={x: batch_x, Y_: batch_y})
        if (i%50==0):
            print("Iteration {0}, loss = {1}".format(i, t_loss))

    # Saving the model
    save_path = saver.save(sess, PATH + "my_mnist.ckpt")
    print("Model saved in file: {0}".format(save_path))

    # Testing
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(acc, feed_dict={x: mnist_data.test.images, Y_: mnist_data.test.labels}))

    return

if (__name__=="__main__"):
    main()
