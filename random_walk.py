import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

def random_walk(M1, t=5, gamma=0.55):
    K = rbf_kernel(M1, gamma)

    D = np.diag([1/np.sum(row) for row in K])
    D2 = np.diag([1/np.math.sqrt(x) for x in np.diag(D)])
    L = D2 @ K @ D2

    L_tilde = np.linalg.matrix_power(L, t)
    D_tilde = np.diag([np.math.sqrt(1/x) for x in np.diag(L_tilde)])

    D1 = np.diag([np.math.sqrt(x) for x in np.diag(D)])
    P_t = np.linalg.matrix_power((np.linalg.inv(D) @ K), t)

    K_tilde = D_tilde @ D1 @ P_t @ D2 @ D_tilde

    return K_tilde