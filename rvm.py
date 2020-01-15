import numpy as np
from scipy.special import expit
from scipy.optimize import minimize
from sklearn.base import ClassifierMixin

def RVC(K, t):
    # Assume zero-mean gaussian prior
    w_mean = np.zeros(K.shape[1])
    
    # Alpha is vector of precison parameters for w
    # Initialize to random values
    alpha = np.random.rand(w_mean.shape[1])

    # y is the logistic sigmoid function
    y = expit(np.dot(w_mean, K.T))

    # Re-estimating alpha
    for i in range(3000):
        # Not possible to find analytical posterior, approximate using Laplace approximation
        # w_star is the posterior mean, sigma the posterior covariance
        A = np.diag(alpha)
        B = np.diag([y_n * (1 - y_n) for y_n in y])
        w_star = np.linalg.inv(A) @ K.T @ (t - y)
        sigma = np.linalg.inv(K.T @ B @ K + A)

        alpha_new = [(1 - alpha[i] * sigma[i, i])/w_star[i] ** 2 for i in range(len(alpha))]

        delta = np.amax(np.absolute(alpha_new - alpha))
        
        if delta < 1e-3:
            break

        alpha = alpha_new
    
    return alpha_new








