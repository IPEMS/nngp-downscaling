# utilites

# for matrix
import numpy as np

# for GPU accel
from numba import njit

# for Fortran funcs
from fortran_py import daxpy, dtrmv

@njit
def logitInv(z, a, b):
    return b-(b-a)/(1+np.exp(z))

@njit 
def logit(theta, a, b):
    return np.log((theta-a)/(b-theta))

@njit
def mvrnorm(des, mu, cholCov, dim):
    # get rvariables

    des = np.random.normal(loc = 0, scale = 1, size = dim)

    # convert cholCov to n x n
    cholCov_2D = cholCov.reshape(dim, dim)

    #########
    # dtrmv #
    #########

    des = dtrmv(cholCov_2D, des)

    #########
    # daxpy #
    #########

    des = daxpy(des, 1, mu)

    return des

@njit
def Q(B, F, u, v, n, nnIndx, nnIndxLU):
    q = 0

    for i in range(0, n, 1):
        a = 0
        b = 0
        for j in range(0, int(nnIndxLU[n + i]), 1):
            a += B[int(nnIndxLU[i] + j)] * u[int(nnIndx[int(nnIndxLU[i] + j)])]
            b += B[int(nnIndxLU[i] + j)] * v[int(nnIndx[int(nnIndxLU[i] + j)])]
        q +=(u[i] - a) * (v[i] - b)/F[i]

    return q

@njit
def dist2(a1, a2, b1, b2):
    return np.sqrt((a1-b1)**2 + (a2-b2)**2)

@njit 
def spCor(D, phi):
    return float(np.exp(-phi*D))