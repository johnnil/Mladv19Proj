import os
from pathlib import Path

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


def read_data_folder(folder):
    x = []
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r', encoding='latin-1') as f:
            x.append(f.read())
    return x


def get_data():
    par_dir_list = os.listdir(Path().parent.absolute())
    if '20news-18828' not in par_dir_list:
        if '20news-18828.tar.gz' in par_dir_list:
            raise FileNotFoundError('Folder 20news-18828 not found. You need to unzip 20news-18828.tar.gz')
        else:
            raise FileNotFoundError('Folder 20news-18828 not found.')

    x_mac = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.sys.mac.hardware'))
    x_win = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.windows.x'))
    data = x_mac + x_win

    y_mac = np.zeros(len(x_mac))
    y_win = np.ones(len(x_win))
    y = np.hstack((y_mac, y_win))

    vect = TfidfVectorizer(min_df=4, smooth_idf=False, encoding='latin-1', norm=None)
    x = vect.fit_transform(data)
    return x, y


if __name__ == '__main__':
    x, y = get_data()
