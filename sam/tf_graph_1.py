'''Test out the idea from Sam book
The graph
'''

import tensorflow as tf

a = tf.constant(5, name="input_a")
b = tf.constant(3, name="input_b")
c = tf.multiply(a, b, name="mul_c")
d = tf.add(a, b, name="add_d")
e = tf.add(c, d, name="add_e")

sess = tf.Session()
print(sess.run(c))

# write graph to tensorboard
writer = tf.summary.FileWriter('./my_graph', sess.graph)
writer.close()
sess.close()
