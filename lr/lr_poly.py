import numpy as np
import matplotlib.pyplot as plt

# Load the data
X = []
Y = []
for line in open("data_poly.csv"):
    [x,y] = line.split(",")
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# plot the data
plt.scatter(X[:,1], Y)
plt.show()

# Calculate weights
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_prediction = np.dot(X, w)

# plot the data with curve
plt.scatter(X[:,1], Y)
plt.plot(sorted(X[:,1]), sorted(Y_prediction))
plt.show()

# Calculate r-squared
d1 = Y - Y_prediction
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print("R-squared : {0}".format(r2))
