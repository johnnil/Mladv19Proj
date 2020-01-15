
import numpy as np
import gaussian_mixture
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def fisher_kernel():
    return 0

def marginalized_kernel(x, k=2):
    """
    :param x:   Data. dim = N x n
    """

    y = x
    #sum( p(k | x) * p(k | y) * x.T * sigma_k * y)
    # p(k | '') is posterior and Sigma_k is variance or covariance.
    # Check what posterior.var() dimensions are and that they are correct with a test case
    # kxy = np.sum(posterior.pdf(x) * posterior.pdf(y) * x.T * posterior.var() * y, axis=-1)

    #est_means, est_var, est_weights = em_gaussian_mix(x, k) #Run em-algorithm
    gmm = GaussianMixture(n_components=k, covariance_type='full', reg_covar=0.01, max_iter=30)
    gmm.fit(x)
    est_means = gmm.means_
    est_var = gmm.covariances_
    est_weights = gmm.weights_
    print("Finished EM!")

    kxy = 0.0
    for i in range(k):
        test1 = multivariate_normal.pdf(x, mean=est_means[i], cov=est_var[i])
        test2 = multivariate_normal.pdf(y, mean=est_means[i], cov=est_var[i])
        test3 = (x @ np.linalg.inv(est_var[i]) @ y.T)
        kxy = kxy + est_weights[i] * (multivariate_normal.pdf(x, mean=est_means[i], cov=est_var[i])[:, None]\
                                    @ multivariate_normal.pdf(y, mean=est_means[i], cov=est_var[i])[:, None].T)\
                                    * (x @ np.linalg.inv(est_var[i]) @ y.T)

    return kxy

def em_gaussian_mix(D, k):
    """

    :param D: Data points. Dim = N x n, N is number of data points n is dimension of data
    :param k: number of gaussians.
    :return: Parameters for gaussian mixtures.
    """
    TOL = 0.00001 # Convergence

    N = D.shape[0]

    #Initialize parameters. everything is just guessed to be same.
    vars = np.array([[[1.0, 0.0], [0.0, 1.0]]] * k)#(Co)Var. dim = k x n x n

    #Use K-Means to initialize.....
    kmeans = KMeans(n_clusters=k, max_iter=20).fit(D)
    unique, counts = np.unique(kmeans.labels_, return_counts=True)
    indices = np.array([ [np.argwhere(kmeans.labels_ == i).squeeze()] for i in unique]).squeeze()

    means = kmeans.cluster_centers_                             #Init means
    vars = np.array([np.cov(D[indices[i]].T) for i in unique])  #Init vars
    weights = counts/N                                          #Init weights

    gamma = np.zeros(shape=(k, N))  # dim = k x n
    log_lik = -10.0                 # log likelihood
    prev_log_lik = -10.

    while(True):
        #E-step: Eval. responsibilities
        for i in range(k):
            gamma[i, :] = weights[i] * multivariate_normal.pdf(D, means[i], vars[i])
        #normalize
        numer = np.sum(gamma, axis=0)
        gamma = gamma/ numer
        Nk = np.sum(gamma, axis=1)

        #M step: re-estimate paramters
        for i in range(k):
            means[i] = np.sum(gamma[i,:,None] * D, axis=0) / Nk[i] #new mean
            vars[i] = (D - means[i]).T @ (gamma[i,:,None] * (D - means[i])) / Nk[i] #new (Co)Var
        #new weights
        weights = Nk / N

        #Evaluate log likelihood
        for i in range(k):
            log_lik = np.log(weights[i] * multivariate_normal.pdf(D, means[i], vars[i]))
        log_lik = np.sum(log_lik)
        diff = np.abs(prev_log_lik - log_lik)
        prev_log_lik = log_lik
        #Check for convergence, continue till convergence
        if diff < TOL:
            break

    return means, vars, weights

def main():
    import matplotlib.pyplot as plt

    print("Hellow World!")

    mean = np.array([[3.0, 0.0], [4.0, -2.0], [3.5, 2.0]])
    var = np.array([[[0.5, 0.5], [0.5, 1.0]], [[0.5, 0.0], [0.0, 1.0]], [[0.25, 0.0], [0.0, 0.25]]])
    weights = np.array([0.333333, 0.33333333, 0.3333333])
    samples, I = gaussian_mixture.sample(weights, mean, var, 1000)

    print("Test with a gaussian mixture")
    est_means, est_var, est_weights = em_gaussian_mix(samples, 3) #Run em-algorithm
    X, Y = np.mgrid[1:6:.01, -4.5:4:.01]
    pos = np.dstack((X, Y))
    prev_k = 0
    for i, k in enumerate(I[0]):
        x, y = samples[prev_k:prev_k + k, 0], samples[prev_k:prev_k + k, 1]
        plt.contour(X, Y, multivariate_normal.pdf(pos, mean=est_means[i], cov=est_var[i]))
        plt.plot(x, y, '.', label="gauss k =" + str(i))
        prev_k = k + prev_k
    # print(samples)
    plt.legend()
    plt.show()

    K = marginalized_kernel(samples, k=3)
    print("Done!")

if __name__ == "__main__":
    main()