import numpy as np
import matplotlib.pyplot as plt

def sample(weights, mean, var, sample_size):
    """

    :param weights: probabilities for each gaussian. Dim = k x 1. Sum needs to be 1.
    :param mean: mean for the gaussians. Dim = k x n
    :param var: var for the gaussians. Dim = k x n x n <- covariance if n > 1
    :param sample_size: number of samples to draw.
    :return: return vector with Dim = sample_size x n, optional: return the num of samples
    """

    try:
        if np.abs(np.sum(weights) - 1.0) > 0.000001:
            raise Exception("Weights for gaussian mixture do not add to 1:", weights)
        if mean.shape[0] != var.shape[0]:
            raise Exception("Mean and (Co)Variance dimension missmatch.", mean.shape[0], "vs. ", var.shape[0])
    except Exception as e:
        print(e)
        exit(1)

    #set correct n:
    n = 0
    if mean.shape[0] == mean.size:
        n = 1
    else:
        _, n = mean.shape

    #sample from i'th gaussian with cat. dist. sample:
    I = np.random.multinomial(sample_size, weights, size=1)
    print(I)
    #Sample point from chosen gaussian
    samples = np.zeros(shape=(sample_size, n))
    prev_k = 0
    for i, k in enumerate(I[0]):
        print(mean[i])
        if n is 1:
            samples[prev_k:prev_k+k, :] = np.random.normal(mean[i], var[i], k)[:, None]
        else:
            samples[prev_k:prev_k + k, :] = np.random.multivariate_normal(mean[i], var[i], k)[:]
        prev_k = k + prev_k
        #print(k)

    #print(samples)
    return samples, I


def main():
    print("Hello World!")
    #sample(np.array([0.1,  0.5, 0.4]), np.array([0.0, 1.0, -1.0]), np.array([1.0, 0.5, 0.25]), 100)

    #Print out a test case:
    mean = np.array([[3.0, 0.0], [4.0, -2.0], [3.5, 2.0]])
    cov = np.array([[[0.5, 0.5], [0.5, 1.0]], [[0.5, 0.0], [0.0, 1.0]], [[0.25, 0.0], [0.0, 0.25]]])
    weights = np.array([0.333333,  0.33333333, 0.3333333])
    samples, I = sample(weights, mean, cov, 1000)

    prev_k = 0
    for i, k in enumerate(I[0]):
        x, y = samples[prev_k:prev_k+k, 0], samples[prev_k:prev_k+k, 1]
        plt.plot(x, y,'.', label="gauss k =" + str(i))
        prev_k = k + prev_k
    #print(samples)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()