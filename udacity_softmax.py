"""
Udacity - Deep learning nano course
Module 1
Softmax.
"""

import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def main():
    scores = np.array([3.0, 1.0, 0.2])
    print(softmax(scores))
    print(softmax(scores * 10))
    print(softmax(scores / 10))

    # Plot softmax curves
    x = np.arange(-2.0, 6.0, 0.1)
    scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

    plt.plot(x, softmax(scores / 10).T, linewidth=2)
    plt.show()
    return

if __name__ == "__main__":
    main()
