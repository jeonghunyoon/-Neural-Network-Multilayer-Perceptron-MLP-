import random

import numpy as np


def sigmoid(a):
    z = 1.0 / np.exp(-1 * a)
    return z


# input : a, ouput : model's output
def feed_forward(self, z):
    for b, w in zip(self.biases, self.weights):
        # multiplication of matrix w and input vector z
        a = np.dot(w, z) + b
        z = sigmoid(a)
    return z


# stochastic gradient descent
def gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    # after each epochs, model will be tested partially.
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for i in xrange(epochs):
        random.shuffle(training_data)
        mini_batches = [training_data[k:k + mini_batch_size] for k in xrange(0, n, mini_batch_size)]
        # on-line gradient descent(one data point)  => mini-batch gradient descent
        for mini_batch in mini_batches:
            self.update_mini_batch(eta, mini_batch)
        if test_data:
            print "Epochs {0}: {1} / {2}".format(i, self.evaluate(test_data), n_test)
        else:
            print "Epochs {0} complete.".format(i)
