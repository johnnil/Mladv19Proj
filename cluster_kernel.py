"""
This file is an implementation of the cluster kernel described in section 3 of the source material
(2257-cluster-kernels-for-semi-supervised-learning.pdf)
"""

import numpy as np
from numpy.linalg import eig
from sklearn.metrics.pairwise import rbf_kernel
#import sys
#np.set_printoptions(threshold=sys.maxsize)

t = 5  #1-5


def kernel(X, k=0, kernel="linear", n=0):
    """
    1. As before, compute the RBF matrix K from both labeled and unlabeled
    points (this time with 1 on the diagonal and not 0) and D, the diagonal
    matrix whose elements are the sum of the rows of K.
    """
    # Affinity matrix using RBF kernel
    A = rbf_kernel(X, gamma=0.55)
    # Let D be a diagonal matrix with diagonal elements equal to the sum of the rows (or the columns)
    D = np.diag(A.sum(axis=1))

    print("check step 1")
    """
    2. Compute L and its eigendecomposition L = UAU^T.
    """
    # Compute L
    Dspecial = np.linalg.inv(D ** (0.5))
    L = Dspecial @ A @ Dspecial
    print("check step 2")

    """
    3. Given a transfer function phi, let ytilde_i = phi(y_i), where the y_i are the eigenvalues
    of L, and construct Ltilde = U(Atilde)U^T.
    """
    # Y = upper case lambda, vector of eigenvalues
    if kernel == "linear":
        Ltilde = linear(L)
    elif kernel == "step":
        Ltilde = step(L, 10)
    elif kernel == "linearStep":
        Ltilde = linearStep(L, 10)
    elif kernel == "polynomial":
        Ltilde = polynomial(L)
    elif kernel == "polyStep":
        Ltilde = polyStep(L, n)

    print("check step 3")

    """
    4. Let Dtilde be a diagonal matrix with Dtilde_ii = 1 / Ltilde_ii and compute Ktilde = Dtilde^(1/2) LiJ1/2. 
    """
    Dtilde = np.diag(1 / np.diag(Ltilde))
    Ktilde = (Dtilde ** (0.5)) @ Ltilde @ (Dtilde ** (0.5))

    print("check step 4")
    return Ktilde


"""
Can be of 4 different types:
    - Linear
    - Step
    - Linear-step
    - Polynomial
"""
def linear(L):
    Ltilde = L
    print("Linear complete")
    return Ltilde

def step(L, k):
    # Eigendecomposition
    Y, U = eig(L)
    ycut = Y[Y.argsort()[-k]]
    Ytilde = [1 if y >= ycut else 0 for y in Y]
    Ltilde = U @ np.diag(Ytilde) @ np.transpose(U)
    print("Step complete")
    return Ltilde

def linearStep(L, k):
    # Eigendecomposition
    Y, U = eig(L)
    ycut = Y[Y.argsort()[-k]]
    Ytilde = [y if y >= ycut else 0 for y in Y]
    Ltilde = U @ np.diag(Ytilde) @ np.transpose(U)
    print("Linear-step complete")
    return Ltilde

def polynomial(L):
    Ltilde = np.linalg.matrix_power(L, t)  # Should be as fast as **
    print("Polynomial complete")
    return Ltilde

def polyStep(L, n):
    # Eigendecomposition
    Y, U = eig(L)
    Ytilde = []
    for i, y in enumerate(Y):
        Ytilde.append((Y[i] ** 0.5) if (i <= n+10) else (Y[i] ** 2))
    #Ytilde = [(y ** 0.5) if (i <= n+10) else (y ** 2) for y in Y]
    Ltilde = U @ np.diag(Ytilde) @ np.transpose(U)
    print("Polynomial-step complete")
    return Ltilde