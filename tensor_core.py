'''
Learning tensor CORE
'''

import tensorflow as tf
import numpy as np

def main():
    hello = tf.constant("Hello, Tensorflow!")
    sess = tf.Session()
    print(sess.run(hello))
    return

if __name__ == "__main__":
    main()
