# *-* coding: utf-8 *-

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def read_mnist():

    mnist = input_data.read_data_sets('MNIST_data/', one_hot=True, reshape=False)

    X_train, Y_train = mnist.train.images, mnist.train.labels
    X_validation, Y_validation = mnist.validation.images, mnist.validation.labels
    X_test, Y_test = mnist.test.images, mnist.test.labels

    # Pad images with 0s (converts 28x28x1 to 32x32x1).
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test
