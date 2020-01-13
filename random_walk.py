import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
import sys
np.set_printoptions(threshold=sys.maxsize)

import os
from pathlib import Path
from sklearn.svm import SVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

def random_walk(x_labeled, x_unlabeled, x_test, y_test, l, sigma=0.6, K=10, t=8):
    """
    Naive implementation of a Markov Random Walk Classifier
    for partially labeled data, as proposed by 
    Martin Szummer and Tommi Jakola 
    """

    print("Classifying points with a naive markov random walk")

    X = np.vstack((x_labeled, x_unlabeled, x_test))

    # Create K-nearest-neighbor graph
    G = kneighbors_graph(X, n_neighbors=K, include_self=True)

    # Symmetrize G
    G = G.toarray()
    G = G + G.T
    
    # I'm sorry for this
    G = np.array([[1 if x == 2 else x for x in row] for row in G])

    # W is the weight matrix
    W = -euclidean_distances(X)
    W = W/sigma
    W = np.exp(W)
    W = W * G

    # Create transition matrix A
    W_sum = np.sum(W, axis=1)
    A = W / W_sum[:, None]

    # Check that A is row-stochastic
    # print(np.sum(A, axis=1))
    # assert(all(np.equal(np.sum(A, axis=1), 1)))

    # Conditional probablities after time t
    A = np.linalg.matrix_power(A, t)

    # Flip the conditional
    P_i = np.ones(len(X)) / len(X)
    P_i_given_k = P_i[:, None] * A

    # Normalize
    P_i_given_k = P_i_given_k / np.sum(P_i_given_k, axis=1)[:, None]

    # Estimate probability of class mac given original starting point of walk
    # Test points are interpreted as a sample of the walk
    #P = expectation_maximization(P_i_given_k, l)

    # Classify point on likelihood of coming from mac
    P_post = np.sum(P_i_given_k[:, :l], axis=1) - np.sum(P_i_given_k[:, l:2*l], axis=1)

    # Posterior is calculated as P(y|k) = P(y|i) * sum(P(i|k))
    #P_post = P * np.sum(P_i_given_k, axis=1)

    # Prediction time
    pred = np.array([-1 if val > 0 else 1 for val in P_post])
    acc = np.sum(pred[-len(y_test):] == y_test) / len(y_test)
    return acc

def expectation_maximization(A, l):
    """
    Implementaion of the EM-algorithm
    """

    print("Starting to maximize expectation")

    # Estimate of each point being mac
    # Start estimate of unlabeled points is 0.5
    P = np.ones(len(A)) * 0.5

    # EM until convergence with limit 3000
    for i in range(3000):

        print(i)

        # E-step
        P_k = P[:, None] * A

        # Normalize
        P_k = P_k / np.sum(P_k, axis=1)[:, None]

        # M-step
        P_new = np.sum(P_k[:, :l], axis=1) / np.sum(P_k, axis=1)

        # Check for convergence
        delta = np.amax(np.absolute(P - P_new))
        if delta < 1e-3:
            break

        # Update Values
        P = P_new

    return P

def classify(x_labeled, x_unlabeled, x_test, y_labeled, y_test):
    """
    Classification algorithm using markov random walks
    Reliant on a lot of things, will fix if necessary
    """

    print("The race is on")

    X = np.vstack((x_labeled, x_unlabeled, x_test))
    P_post = random_walk(X, l)
    
    # Prediction time
    pred = np.array([-1 if val > 0 else 1 for val in P_post[2*l:]])
    acc = np.sum(pred == y_test) / len(y_test)
    return acc








# def read_data_folder(folder):
#     x = []
#     for file_name in os.listdir(folder):
#         with open(os.path.join(folder, file_name), 'r', encoding='latin-1') as f:
#             x.append(f.read())
#     return x


# def get_data():
#     par_dir_list = os.listdir(Path().parent.absolute())
#     if '20news-18828' not in par_dir_list:
#         if '20news-18828.tar.gz' in par_dir_list:
#             raise FileNotFoundError('Folder 20news-18828 not found. You need to unzip 20news-18828.tar.gz')
#         else:
#             raise FileNotFoundError('Folder 20news-18828 not found.')

#     # read data strings (hopefully this works on all OS's and in jupyter)
#     x_mac = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.sys.mac.hardware'))
#     x_win = read_data_folder(os.path.join(Path().absolute(), '20news-18828', 'comp.windows.x'))
#     data = x_mac + x_win

#     # data labels for each class
#     y_mac = -np.ones(len(x_mac))
#     y_win = np.ones(len(x_win))

#     # 3 or less occurances means a words is removed
#     vect = TfidfVectorizer(min_df=4, smooth_idf=False, encoding='latin-1')
#     vect.fit(data)
#     x_mac = vect.transform(x_mac).toarray()
#     x_win = vect.transform(x_win).toarray()

#     # shuffle each class independently
#     i = np.random.permutation(np.arange(x_mac.shape[0]))
#     j = np.random.permutation(np.arange(x_win.shape[0]))
#     return x_mac[i], x_win[j], y_mac[i], y_win[j]

# if __name__ == '__main__':
#     x_mac, x_win, y_mac, y_win = get_data()
#     l = 50
#     x_labeled = np.vstack((x_mac[:l], x_win[:l]))
#     x_unlabeled = np.vstack((x_mac[l:], x_win[l:]))
#     y = np.hstack((y_mac[:l], y_win[:l]))
#     y_test = np.hstack((y_mac[l:], y_win[l:]))

#     print(y_test.shape)

#     acc = classify(x_labeled, x_unlabeled, y_test, l)
#     print(acc)
