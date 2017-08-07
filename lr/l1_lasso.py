'''
'''

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50

X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D-3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

# Gradient descent
cost = []
w = np.random.randn(D)
learning_rate = 0.001
l1 = 10.0
for t in xrange(500):
    Y_prediction = X.dot(w)
    delta = Y_prediction - Y
    w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

    mse = delta.dot(delta) / N
    cost.append(mse)

# Plot the cost
plt.plot(cost)
plt.show()

print("final w {0}".format(w))

plt.plot(true_w, label="true_w")
plt.plot(w, label="map_w")
plt.legend()
plt.show()
