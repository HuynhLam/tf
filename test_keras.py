''' Test karas on MNIST
'''

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
import os

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = "2"
    # Import the MNIST data
    #mnist = input_data.read_data_sets('mnist')
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess input data
    X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Preprocess class labels
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28), dim_ordering='th'))
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
    print("******* Training: run 10 epochs... *******")
    model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=1)

    # Evaluation
    print("******* Testing *******")
    loss_and_metrics = model.evaluate(X_test, Y_test, verbose=1)

    return

if __name__ == '__main__':
    main()
