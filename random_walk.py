import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import kneighbors_graph
import sys
np.set_printoptions(threshold=sys.maxsize)

def random_walk(x_labeled, x_unlabeled, x_test, y_labeled, y_test, sigma=0.6, K=10, t=8):
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
    P_post = np.sum(P_i_given_k[:, :len(y_labeled)] * y_labeled, axis=1)

    # Posterior is calculated as P(y|k) = P(y|i) * sum(P(i|k))
    #P_post = P * np.sum(P_i_given_k, axis=1)

    # Prediction time
    pred = np.array([0 if val < 0 else 1 for val in P_post])
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