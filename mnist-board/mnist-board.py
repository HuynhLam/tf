import os
import os.path
import shutil
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.learn as learn
import keras

from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D

# run on 'gpu:2'
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# For embedding GUI
LABELS = os.path.join(os.getcwd(), "labels_1024.tsv")
SPRITES = os.path.join(os.getcwd(), "sprite_1024.png")
# Load mnist datasets
mnist_data = learn.datasets.mnist.read_data_sets("mnist", one_hot=True)

# callbacks
callbacks = keras.callbacks.TensorBoard(log_dir='./graph1/kr,conv=2,fc=2', write_graph=True)


def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        act = tf.nn.relu(conv + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


def fc_layer(input, size_in, size_out, name="fc"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        act = tf.matmul(input, w) + b
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        return act


def mnist_model(learning_rate, use_two_fc, use_two_conv, hparam):
    tf.reset_default_graph()
    sess = tf.Session()

    # Setup placeholders, and reshape the data
    x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', x_image, 3)
    y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

    if use_two_conv:
      conv1 = conv_layer(x_image, 1, 32, "conv1")
      conv_out = conv_layer(conv1, 32, 64, "conv2")
    else:
      conv1 = conv_layer(x_image, 1, 64, "conv")
      conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64])


    if use_two_fc:
      fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1")
      relu = tf.nn.relu(fc1)
      embedding_input = relu
      tf.summary.histogram("fc1/relu", relu)
      embedding_size = 1024
      logits = fc_layer(fc1, 1024, 10, "fc2")
    else:
      embedding_input = flattened
      embedding_size = 7*7*64
      logits = fc_layer(flattened, 7*7*64, 10, "fc")

    with tf.name_scope("xent"):
      xent = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(
              logits=logits, labels=y), name="xent")
      tf.summary.scalar("xent", xent)

    with tf.name_scope("train"):
      train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

    with tf.name_scope("accuracy"):
      correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      tf.summary.scalar("accuracy", accuracy)

    summ = tf.summary.merge_all()


    embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
    assignment = embedding.assign(embedding_input)
    saver = tf.train.Saver()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("graph1/" + hparam)
    writer.add_graph(sess.graph)

    config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding.name
    embedding_config.sprite.image_path = SPRITES
    embedding_config.metadata_path = LABELS
    # Specify the width and height of a single thumbnail.
    embedding_config.sprite.single_image_dim.extend([28, 28])
    tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

    for i in range(300001):
        batch = mnist_data.train.next_batch(32)
        if i % 5 == 0:
            [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]})
            writer.add_summary(s, i)
        if i % 500 == 0:
            sess.run(assignment, feed_dict={x: mnist_data.test.images[:1024], y: mnist_data.test.labels[:1024]})
            saver.save(sess, os.path.join("graph1/", "model.ckpt"), i)
        sess.run(train_step, feed_dict={x: batch[0], y: batch[1]})

def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def keras_mnist_model():
    # Load MNIST data from keras, different from tensorflow, 60k train + 10k test
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Map test data vector to labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    #  Build the model
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), dim_ordering='tf'))
    model.add(Convolution2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile the model for use
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Training
    print("******* Training: run 5 epochs... *******")
    model.fit(X_train, Y_train, batch_size=32, epochs=5, verbose=1, callbacks = [callbacks])

    # Evaluation
    print("******* Testing *******")
    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=1)
    print("\ntest_result: {0}".format(loss_and_metrics))


def main():
    print("Modified2")

    # Run keras model
    keras_mnist_model()

    for learning_rate in [1E-3, 1E-4, 1E-5]:

        # Trial-error modul architectures
        for use_two_fc in [True]:
            for use_two_conv in [True, False]:
                start_time_begin = time.time()
                # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2")
                hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
                print("***** Starting run model {0} *****".format(hparam))
                # Actually run with the new settings
                mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)
                print("***** Training time {0:.2f} s *****".format(time.time()-start_time_begin))

    print("Done ! Lets see the board.")

if __name__ == '__main__':
    main()
