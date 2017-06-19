''' Create variables, which is mutable tensor values
Variables handle by session, so we must init them inside a Session
'''

import tensorflow as tf
import numpy as np

my_var = tf.Variable(1, name="my_variable")

add = tf.add(5, my_var)
mul = tf.multiply(3, my_var)

# we can use helper ops to create common tensor type
# 2x2 matrix of zeros
zeros = tf.zeros([2, 2])

# vector with length = 6 of ones
ones = tf.ones([6])

# 3x3x3 Tensor of random uniform values between 0 to 10
uniform = tf.random_uniform([3,3,3], minval=0, maxval=10)

# 3x3x3 Tensor of normal distribution values, mean = 0, standard deviation = 2
normal = tf.random_normal([3,3,3], mean=0.0, stddev=2.0)

# 2x2 Tensor of distribution without values <3 and >7
trunc = tf.truncated_normal([2,2], mean=5.0, stdded=1.0)

# Fast declaration
random_var = tf.Variable(tf.truncated_normal([2,2]))

# Initialize all the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Initialize a subset of variables
var1 = tf.Variable(0, name="initizlize_this")
var2 = tf.Variable(1, name="leave_this_alone")
init1 = tf.initialize_variables([var1], name="init_var1")
sess.run(init1)
