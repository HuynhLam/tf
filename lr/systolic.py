'''
The data file "mlr02.xls" containt:
X1 = systolic blood pressure
X2 = age in years
X2 = weights in pounds
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load the data
df = pd.read_excel('mlr02.xls')
X = df.as_matrix()

# see the data
plt.scatter(X[:,1], X[:,0])
plt.show()

plt.scatter(X[:,2], X[:,0])
plt.show()

# group data
#df["ones"] = 1
df['ones'] = np.random.rand(11)
df.info()
print(df.head())
Y = df["X1"]
onlyX2 = df[["X2", "ones"]]
onlyX3 = df[["X3", "ones"]]
bothX = df[["X2", "X3", "ones"]]

#print(bothX.shape)

def get_r2(X, Y):
    w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
    Y_prediction = np.dot(X, w)

    d1 = Y - Y_prediction
    d2 = Y - Y.mean()
    return (1 - d1.dot(d1) / d2.dot(d2))

'''
Calculate r-squared for 3 cases:
+ Using only feature X2
+ Using only feature X3
+ Using both features X2 and X3
'''
print("Calculate r-squared \nFeature X2: {0}".format(get_r2(onlyX2, Y)))
print("Feature X3: {0}".format(get_r2(onlyX3, Y)))
print("Features X2 & X3: {0}".format(get_r2(bothX, Y)))
