import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelPropagation
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

import cluster_kernel
import clustered_representation
import random_walk
import usps


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
    x = np.vstack((x_labeled, x_unlabeled, x_test))
    k = kernel(x)
    k_ = k[:len(x_labeled), -len(x_test):]
    y_prediction = np.where((y @ k_) >= 0, 1, -1.)
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc

def evaluate_kernel_2(x_label_i, x_test, y, y_test, k):
    """
    :param x_label_i: label indices
    :param x_test_i: test indices
    :param y: labels for training points
    :param y_test: labels for test points
    :param k: kernel values.
    :return:
    """
    #x = np.vstack((x_labeled, x_unlabeled, x_test))
    #k = kernel(x)
    k_ = k[x_label_i, -len(x_test):]
    y_prediction = np.where((y @ k_) >= 0, 1, -1.)
    acc = np.sum(y_prediction == y_test) / len(y_test)
    return acc


def evaluate_kernel_SVM(x_labeled, x_unlabeled, x_test, y, y_test, kernel):
    """
    Same parameters as `evaluate_kernel` but uses SVC.
    """
    n_train = len(x_labeled)
    n_test = len(x_test)
    x = np.vstack((x_labeled, x_unlabeled, x_test))
    k = kernel(x)

    svc = SVC(C=5, kernel='precomputed')
    svc.fit(k[:n_train, :n_train], y)
    y_prediction = svc.predict(k[-n_test:, :n_train])
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

    # read data strings (hopefully this works on all OS's and in jupyter)
    x_mac = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.sys.mac.hardware'))
    x_win = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.windows.x'))
    data = x_mac + x_win

    # data labels for each class
    y_mac = -np.ones(len(x_mac))
    y_win = np.ones(len(x_win))

    # 3 or less occurances means a words is removed
    vect = TfidfVectorizer(min_df=4, smooth_idf=False, encoding='latin-1')
    vect.fit(data)
    x_mac = vect.transform(x_mac).toarray()
    x_win = vect.transform(x_win).toarray()

    # shuffle each class independently
    i = np.random.permutation(np.arange(x_mac.shape[0]))
    j = np.random.permutation(np.arange(x_win.shape[0]))
    return x_mac[i], x_win[j], y_mac[i], y_win[j]


def perform_test(kernel, l=8):
    """
    Perform test according to 4.3 for a given kernel.

    :param l : number of labels to use
    :return: accuracy of the model
    """
    np.random.seed(424242)  # reproducibility
    x_mac, x_win, y_mac, y_win = get_data()
    x_test = np.vstack((x_mac[-500:], x_win[-500:]))
    y_test = np.hstack((y_mac[-500:], y_win[-500:]))
    x_mac, x_win, y_mac, y_win = x_mac[:-500], x_win[:-500], y_mac[:-500], y_win[:-500]

    #x_mac_i = np.arange(x_mac.shape[0])
    #x_win_i = np.arange(x_win.shape[0])

    #x_labeled = np.vstack((x_mac[:l], x_win[:l]))
    #x_unlabeled = np.vstack((x_mac[l:], x_win[l:]))
    #x = np.vstack((x_labeled, x_unlabeled, x_test))
    #k = kernel(x)

    #l = 8
    acc = [None] * 10
    for test in range(10):
        np.random.shuffle(x_mac)
        np.random.shuffle(x_win)
        #x_labeled_i = np.hstack((x_mac_i[:l], x_win_i[:l]))
        #x_unlabeled = np.vstack((x_mac_i[l:], x_win_i[l:]))
        x_labeled = np.vstack((x_mac[:l], x_win[:l]))
        x_unlabeled = np.vstack((x_mac[l:], x_win[l:]))

        y_labeled = np.hstack((y_mac[:l], y_win[:l]))

        #acc[test] = evaluate_kernel_2(x_labeled_i, x_test, y_labeled, y_test, k)
        acc[test] = evaluate_kernel(x_labeled, x_unlabeled, x_test, y_labeled, y_test, kernel)
        #acc[test] = random_walk.random_walk(x_labeled, x_unlabeled, x_test, y_labeled, y_test)
    acc = np.array(acc)
    return acc.mean(), acc.std()

def label_experiment(kernels, names=None):
    """

    :param kernel: kernels to perform experiments on.
    :param names: Name for the kernels
    :return: something
    """

    num_l = [2, 4, 8, 16, 32, 64, 128]

    acc_means = np.empty(shape=(len(kernels), len(num_l)))
    acc_stds = np.empty_like(acc_means)

    for i, kernel in enumerate(kernels):
        #Run experiment with different number of label
        print(names[i])
        for j, l in enumerate(num_l):
            print("Labeled points: " + str(l))
            mean_tmp, std_tmp = perform_test(kernel, l)
            acc_means[i][j] = mean_tmp
            acc_stds[i][j] = std_tmp
            print(f'accuracy = {mean_tmp * 100}% (±{std_tmp * 100:.2})')
        
        plt.semilogx(num_l, acc_means[i, :], label=names[i])

    plt.legend()
    plt.savefig('kernelsaccuracy.png')
    plt.show()

def experemint_2(l=8):

    # Experiment comparing random walk, tSVM, SVM and our cluster kernel:
    tSVM = LabelPropagation(max_iter=5000)

    np.random.seed(424242)  # reproducibility
    x_mac, x_win, y_mac, y_win = get_data()
    x_test = np.vstack((x_mac[-500:], x_win[-500:]))
    y_test = np.hstack((y_mac[-500:], y_win[-500:]))
    y_test_tsvm = np.hstack((0.0 * y_mac[-500:], y_win[-500:]))
    x_mac, x_win, y_mac, y_win = x_mac[:-500], x_win[:-500], y_mac[:-500], y_win[:-500]
    y_mac_tsvm = np.zeros((y_mac.shape)) # change -1 to zero
    x_labeled = np.vstack((x_mac[:l], x_win[:l]))
    x_unlabeled = np.vstack((x_mac[l:], x_win[l:]))

    X = np.vstack((x_labeled, x_unlabeled))
    y_labeled = np.hstack((y_mac[:l], y_win[:l]))
    y_labeled_tsvm = np.hstack((y_mac_tsvm[:l], y_win[:l]))
    y_unlabeled = np.hstack((y_mac[l:], y_win[l:]))
    y_unlabeled_tsvm = -np.ones((y_unlabeled.shape)) # Set unlabeled points
    labels_tsvm = np.hstack((y_labeled_tsvm, y_unlabeled_tsvm))

    # Perform tests on tSVM and Random Walk.
    # Cannot use perform test function since data is weird for both.
    acc_tSVM = np.array([None] * 10)
    acc_random_walk = np.array([None] * 10)
    for test in range(10):
        np.random.shuffle(x_mac)
        np.random.shuffle(x_win)
        x_labeled = np.vstack((x_mac[:l], x_win[:l]))
        x_unlabeled = np.vstack((x_mac[l:], x_win[l:]))

        y_labeled = np.hstack((y_mac[:l], y_win[:l]))
        X = np.vstack((x_labeled, x_unlabeled))

        tSVM.fit(X, labels_tsvm)

        acc_tSVM[test] = tSVM.score(x_test, y_test_tsvm)
        print(f'accuracy = {acc_tSVM[test] * 100}% () tSVM')

        acc_random_walk[test] = random_walk.random_walk(x_labeled, x_unlabeled, x_test, y_labeled, y_test)
        print(f'accuracy = {acc_random_walk[test] * 100}% () Random Walk')

        # acc[test] = evaluate_kernel_2(x_labeled_i, x_test, y_labeled, y_test, k)
        #acc[test] = evaluate_kernel(x_labeled, x_unlabeled, x_test, y_labeled, y_test, kernel)
        # acc[test] = random_walk.random_walk(x_labeled, x_unlabeled, x_test, y_labeled, y_test)

    #use perform test function on "normal" SVM & cluster kernel
    kernel1 = lambda x: clustered_representation.kernel(x, 10)
    mean_cluster, std_cluster = perform_test(kernel1, l)
    print(f'tSVM: accuracy = {acc_tSVM.mean() * 100}% (±{acc_tSVM.std() * 100:.2})')
    print(f'random walk: accuracy = {acc_random_walk.mean() * 100}% (±{acc_random_walk.std() * 100:.2})')
    print(f'Cluster kernel: accuracy = {mean_cluster * 100}% (±{std_cluster * 100:.2})')


if __name__ == '__main__':
    # Choose kernel
    kernel1 = lambda x: clustered_representation.kernel(x, 10)
    kernel2 = lambda x: cluster_kernel.kernel(x, 10, "linear", 16)
    kernel3 = lambda x: cluster_kernel.kernel(x, 10, "polynomial", 16)
    kernel4 = lambda x: cluster_kernel.kernel(x, 10, "step", 16)
    kernel5 = lambda x: cluster_kernel.kernel(x, 10, "polyStep", 16)
    #acc_mean, acc_std = perform_test(kernel1)
    #print(f'accuracy = {acc_mean * 100}% (±{acc_std * 100:.2})')
    label_experiment([kernel2, kernel3, kernel4, kernel5], names=["Linear","Polynomial","Step","Polystep"])
    #label_experiment([kernel1], names=["Clustered_kernel","linear","polynomial","step","ploystep"])
    #experemint_2(l = 8)

