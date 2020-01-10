import numpy as np
from sklearn.svm import SVC

import cluster_kernel
import clustered_representation


def evaluate_kernel(x_labeled, x_unlabeled, x_test, y, y_test, kernel):
    """
    Evaluates a kernel with the kernel classification method.
    :param x_labeled: labeled training data points, shape=(n_labeled, d)
    :param x_unlabeled: unlabeled data point, shape=(n_unlabeled, d)
    :param x_test: test data points, shape=(n_test, d)
    :param y: labels for training points, shape=(n_labeled,)
    :param y_test: labels for test points, shape=(n_test,)
    :param kernel: kernel function that should be evaluated, kernel(x) -> array with shape=(n, n)
    :return: fraction of test points that were correctly labelled
    """
    x = np.stack((x_labeled, x_unlabeled, x_test))
    k = kernel(x)
    k_ = k[:len(x_labeled), :len(x_test)]
    y_prediction = (y[None, :] @ k_)[0]
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc


def evaluate_kernel_SVM(x_labeled, x_unlabeled, x_test, y_train, y_test, kernel):
    x = np.stack((x_labeled, x_unlabeled, x_test))
    y = np.stack((y_train, np.zeros(len(x_unlabeled) + len(x_test))))
    w = np.ones_like(y)
    w[len(y_train):] = 0

    svc = SVC(C=0, kernel=kernel)
    svc.fit(x, y_test, w)
    y_prediction = svc.predict(x_test)
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc
