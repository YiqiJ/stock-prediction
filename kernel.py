import numpy as np
import matplotlib.pyplot as plt
import scipy

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.special import kv, gamma

"""
Define Kernel Functions
"""


def weiner(x1, x2):
    K = np.zeros((len(x1), len(x2)))
    for i in range(len(x1)):
        for j in range(len(x2)):
            K[i][j] = min(x1[i][0], x2[j][0])
    return K


def kernel_weiner():
    def kernel_func(x1, x2): return weiner(x1, x2)
    return kernel_func


def rbf(x1, x2, length_scale, sigma_f):
    # distance between each rows
    dist_matrix = np.sum(np.square(x1), axis=1).reshape(-1, 1) + \
        np.sum(np.square(x2), axis=1) - 2 * np.dot(x1, x2.T)
    K = np.square(sigma_f) * np.exp(-1 /
                                    (2 * np.square(length_scale)) * dist_matrix)
    return K


def kernel_rbf(length_scale=1, sigma_f=1):
    def kernel_func(x1, x2): return rbf(x1, x2, length_scale, sigma_f)
    return kernel_func


def matern(x1, x2, length_scale, nu):
    K = cdist(x1 / length_scale, x2 / length_scale, metric='euclidean')
    K[K == 0.0] += np.finfo(float).eps  # strict zeros result in nan
    tmp = (np.sqrt(2 * nu) * K)
    K.fill((2 ** (1. - nu)) / gamma(nu))
    K *= tmp ** nu
    K *= kv(nu, tmp)
    return K


def kernel_matern(length_scale=1.0, nu=1.5):
    """
    nu: The smaller , the less smooth the approximated function is.
    """
    def kernel_func(x1, x2): return matern(x1, x2, length_scale, nu)
    return kernel_func


def RQ(x1, x2, length_scale, alpha):
    d = cdist(x1 / length_scale, x2 / length_scale, metric='euclidean')
    K = (1 + np.square(d)/(2*alpha))**(-alpha)
    return K


def kernel_RQ(length_scale=1.0, alpha=1.0):
    def kernel_func(x1, x2): return RQ(x1, x2, length_scale, alpha)
    return kernel_func

############################### Kernel Operators ###############################


def additive(k1, k2):
    return lambda x1, x2: k1(x1, x2) + k2(x1, x2)


def const(k, c):
    return lambda x1, x2: k(x1, x2)*c


def minus(k1, k2):
    return lambda x1, x2: k1(x1, x2) - k2(x1, x2)
