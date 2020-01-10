import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

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
    x = np.hstack((x_labeled, x_unlabeled, x_test))
    k = kernel(x)
    k_ = k[:len(x_labeled), -len(x_test):]
    y_prediction = (y[None, :] @ k_)[0]
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc


def evaluate_kernel_SVM(x_labeled, x_unlabeled, x_test, y, y_test, kernel):
    """
    Same parameters as `evaluate_kernel` but uses SVC.
    """
    n_train = len(x_labeled)
    n_test = len(x_test)
    x = np.hstack((x_labeled, x_unlabeled, x_test))
    k = kernel(x)

    svc = SVC(C=0, kernel='precomputed')
    svc.fit(k[:n_train, :n_train], y)
    y_prediction = svc.predict(k[-n_test:, -n_test:])
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc


def get_data():
    x_mac, y_mac= fetch_20newsgroups(subset='all', categories=('comp.sys.mac.hardware',),
                                     remove=('headers', 'footers', 'quotes'), return_X_y=True)
    x_win, y_win = fetch_20newsgroups(subset='all', categories=('comp.windows.x',),
                                     remove=('headers', 'footers', 'quotes'), return_X_y=True)
    data = x_mac + x_win
    y_win[:] = 1
    y = np.hstack((y_mac, y_win))

    vect = TfidfVectorizer()
    x = vect.fit_transform(data)
    return x, y


x, y = get_data()
