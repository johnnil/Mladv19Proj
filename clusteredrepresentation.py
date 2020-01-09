import numpy as np
from numpy.linalg import eig, norm
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances


def kernel(x, k):
    n = len(x)
    # x = data[None, :]
    # xx = np.broadcast(.T, x)
    # A = np.empty((n, n))
    # A.flat = [rbf_kernel(xi, xj) for (xi, xj) in xx]
    A = rbf_kernel(x, gamma=0.55)
    A[np.arange(n), np.arange(n)] = 0
    D = np.diag(A.sum(axis=0) ** (-0.5))
    L = D @ A @ D

    # find eigenpairs and take the k biggest ones
    l, v = np.eig(L)
    i = np.flip(l.argsort())
    i = i[:k]

    # create new, normalised representation of data points
    x_ = v[:, i]
    n = norm(x_, axis=1, keepdims=True)
    x_ = x_ / n

    k = euclidean_distances(x_)
    return x_
