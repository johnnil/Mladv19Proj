'''
This file is an implementation of the cluster kernel described in section 3 of the source material
(2257-cluster-kernels-for-semi-supervised-learning.pdf)
'''

import numpy as np
from numpy.linalg import eig
from sklearn.metrics.pairwise import rbf_kernel

def cluster_kernel(X, K):
    """
    1. As before, compute the RBF matrix K from both labeled and unlabeled
    points (this time with 1 on the diagonal and not 0) and D, the diagonal
    matrix whose elements are the sum of the rows of K.
    """
    n = len(X)
    # Affinity matrix using RBF kernel
    A = rbf_kernel(K, gamma=0.55)
    # Set diagonal to 1
    A[np.arange(n), np.arange(n)] = 1
    # Let D be a diagonal matrix with diagonal elements equal to the sum of the rows (or the columns)
    D = np.diag(A.sum(axis=0))

    """
    2. Compute L and its eigendecomposition L = UAU^T.
    """
    # Compute L
    L = (D ** (-0.5)) @ A @ (D ** (-0.5))
    # Eigendecomposition
    Y, U = eig(L)
    # Make U transposable
    U = np.expand_dims(U, axis=1)

    """
    3. Given a transfer function phi, let ytilde_i = phi(y_i), where the y_i are the eigenvalues
    of L, and construct Ltilde = U(Atilde)U^T.
    """
    # Y = upper case lambda, matrix of eigenvalues
    Ytilde = phi(Y)  # alternativt elementvis beroende på hur folk konstruerat transfer functions
    Ltilde = U @ Ytilde @ np.transpose(U)

    """
    4. Let Dtilde be a diagonal matrix with Dtilde_ii = 1 / Ltilde_ii and compute Ktilde = Dtilde^(1/2) LiJ1/2. 
    """
    Dtilde = np.diag(1 / np.diag(Ltilde))

    Ktilde = (Dtilde ** (0.5)) @ Ltilde @ (Dtilde ** (0.5))

    return Ktilde


"""
Can be of 4 different types:
    - Linear
    - Step
    - Linear-step
    - Polynomial
"""
def phi(Y):
    return Y