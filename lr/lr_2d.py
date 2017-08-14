import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Load data
X = []
Y = []
for line in open("data_2d.csv"):
    [x1, x2, y] = line.split(',')
    X.append((1, float(x1), float(x2)))
    Y.append(float(y))

# convert to np.array
X = np.array(X)
Y = np.array(Y)

# plot the data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# Calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_prediction = np.dot(X, w)

# Calculate R-squared
d1 = Y - Y_prediction
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print("R-squared {0}".format(r2))
