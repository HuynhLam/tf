import numpy as np
import tensorflow as tf

a = tf.constant([5, 3], name="input")
b = tf.reduce_prod(a, name="prod_b")
c = tf.reduce_sum(a, name="sum_c")
d = tf.add(b, c, name="add_d")

# example numpy arrays
# 0-D Tensor with 32-bit integer
t_0 = np.array(50, dtype=np.int32)

# 1-D Tensor with byte string
# Note: don't declare dtype when using string in np
t_1 = np.array([b"apple", b"banana", c"cow"])

# 2-D Tensor with  boolean
t_2 = np.array([[True, True, False],
                [True, False, True],
                [True, False, False]], dtype=np.bool)

# 3-D Tensor with 64-bit integer
t_3 = np.array([[ [0, 0], [0, 1], [0, 2] ],
                [ [1, 0], [1, 1], [1, 2] ],
                [ [2, 0], [2, 1], [2, 2] ]], dtype=np.int64)
