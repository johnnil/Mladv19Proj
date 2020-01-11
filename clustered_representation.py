import numpy as np
from numpy.linalg import eig, norm
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def find_gap(l):
    d = l[:-1] - l[1:]
    i = d.argmax()
    return i + 1


def kernel(x, k=0):
    n = len(x)
    # create matrices A, D and L
    A = rbf_kernel(x, gamma=0.55)
    A[np.arange(n), np.arange(n)] = 0
    D = np.diag(A.sum(axis=0) ** (-0.5))
    L = D @ A @ D

    # find eigenpairs and take the k biggest ones
    l, v = eig(L)
    i = np.flip(l.argsort())
    k = k or find_gap(l[i])
    i = i[:k]

    # create new, normalised representation of data points
    x_ = v[:, i]
    n = norm(x_, axis=1, keepdims=True)
    x_ = x_ / n

    k = euclidean_distances(x_)
    return x_
