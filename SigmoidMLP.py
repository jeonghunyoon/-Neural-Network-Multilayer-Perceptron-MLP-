import numpy as np


def sigmoid(a):
    z = 1.0 / np.exp(-1 * a)
    return z


# input : a, ouput : model's output
def feedforward(self, z):
    for b, w in zip(self.biases, self.weights):
        # multiplication of matrix w and input vector z
        a = np.dot(w, z) + b
        z = sigmoid(a)
    return z
