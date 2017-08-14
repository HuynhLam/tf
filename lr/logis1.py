import numpy as np

N = 100
D = 2

X = np.random.randn(N,D)

print(X.shape)
#print(X[:,0])

ones = np.array([[1]*N]).T
#print(ones)

Xb = np.concatenate((ones, X), axis=1)
#print(Xb)

w = np.random.randn(D+1)

z = Xb.dot(w)


def sigmod(z):
    return 1/(1 + np.exp(-z))

print(sigmod(z))
