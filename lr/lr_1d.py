import numpy as np
import matplotlib.pyplot as plt


# Load the data
X = []
Y = []
for line in open("data_1d.csv"):
    [x, y] = line.split(',')
    X.append(float(x))
    Y.append(float(y))


# Convert X and Y to numpy array
print("Type X: {0}".format(type(X)))
print("Type Y: {0}".format(type(Y)))

X = np.array(X)
Y = np.array(Y)

print("Type X: {0}, shape {1}".format(type(X), X.shape))
print("Type Y: {0}, shape {1}".format(type(Y), Y.shape))

# Plot the data

plt.scatter(X,Y)
#plt.axis("equal")
plt.show()

# Apply the equation
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# Calculate the Y prediction line
Y_prediction = a * X + b

# Plot it all
plt.scatter(X, Y)
plt.plot(X, Y_prediction)
plt.show()

# Calculate R-squared
d1 = Y - Y_prediction
d2 = Y - Y.mean()
r2 = 1 - ( d1.dot(d1) / d2.dot(d2) )
print("R-squared: {0:.5f}".format(r2))
