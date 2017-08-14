'''
Demo how tf read csv file
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf


def main():
    filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])

    reader = tf.TextLineReader()
    [key, value] = reader.read(filename_queue)

    record_defaults = [[1], [1], [1], [1], [1]]
    [col1, col2, col3, col4, col5] = tf.decode_csv()

    return

if(__name__=="__main__"):
    main()
