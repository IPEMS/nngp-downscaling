###########################
# Lacpack Fotran Funtions #
###########################

# used for matrix compuatation
import numpy as np

# used to speed up the code
from numba import njit

##########
# dpotrf #
##########

@njit
def dpotrf(length, A):
    # convert from 1D to 2D
    A_orignal = A[:length**2].reshape(length, length) 
    # make the array symetric
    A_sem = np.triu(A_orignal) + np.triu(A_orignal, +1).T
    # run the cholesky factorization
        # gives the upper trinalge
    A_sem = np.linalg.cholesky(A_sem).T # transpose
    A_sem = A_sem + np.tril(A_orignal, -1) # grab the cholesky part and add the lower of the orignal
    A[:length**2] = A_sem.flatten() # return CuPy


##########
# dpotri #
##########

@njit
def dpotri(length, A):
    # convert from 1D to 2D
    A_orignal = A[:length**2].reshape(length, length) 
    # get the upper trinagle (Cholesky Part)
    A_inv = np.triu(A_orignal)

    # get the inverse from the cholesky
    A_inv = np.triu(np.dot(np.linalg.inv(A_inv), np.linalg.inv(A_inv).T))
    A_inv = A_inv + np.tril(A_orignal, -1) # only take the non 0's
    A[:length**2] = A_inv.flatten() # flatten back

#########
# dsymv #
#########

@njit
def dsymv(y, A, x, beta, alpha, length, start):
    # setup
    A_reshape = A[:length**2].reshape(length, length) # trimm A
    x_for_y = x[:length] # trimm x

    A_in_y = np.triu(A_reshape) + np.triu(A_reshape, +1).T # make a symmetric matrix

    # y := alpha*A*x + beta*y
    y_sum = alpha*np.dot(A_in_y, x_for_y)+beta*y[:length]

    # return flatten array
    y[start:(start + len(y_sum))] = y_sum


#########
# Dgemv #
#########

@njit
def dgemv(y, alpha, A, x, beta):
    y = alpha*np.dot(A, x)+beta*y # gets the dot product of A and x and adds it with y times the beta value.
    return y

#########
# Daxpy #
#########

@njit
def daxpy(dy, da, dx):
    dy = dy + da*dx # multiplies 
    return dy

#########
# dtrmv #
#########

@njit
def dtrmv(A, x):
    x = np.dot(A.T, x)
    return x
