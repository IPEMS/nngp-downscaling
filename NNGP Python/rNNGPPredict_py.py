##################
# gets the y hat #
##################

import numpy as np
from numba import njit

from fortran_py import dpotrf, dpotri, dsymv

from util_py import dist2, spCor

import time

@njit
def y_hat_calc(n_samples, theta, nTheta, phiIndx, sigmaSqIndx, tauSqIndx, coords_1D, m, nn_indx_0_1D, i, q, coords_0_1D, C, tmp_m, c, dot_part, dot_part_Y,
          nn_indx_0, Y_1D, zIndx, X_0_1D, n, y_0, z):

    for s in range(0, n_samples, 1):
        # get each phi, sigma, and tau
        phi = theta[s*nTheta+phiIndx]
        sigmaSq = theta[s*nTheta+sigmaSqIndx]
        tauSq = theta[s*nTheta+tauSqIndx]

        # go through nn
        for k in range(0, m, 1):
            d = dist2(coords_1D[nn_indx_0_1D[i+q*k]], coords_1D[n + nn_indx_0_1D[i + q*k]], coords_0_1D[i], coords_0_1D[q + i])
            c[k] = sigmaSq*spCor(d, phi)
            for l in range(0, m, 1):
                d = dist2(coords_1D[nn_indx_0_1D[i+q*k]], coords_1D[n + nn_indx_0_1D[i + q*k]], coords_1D[nn_indx_0_1D[i + q*l]], coords_1D[n + nn_indx_0_1D[i + q*l]])
                C[int(l*m+k)] = sigmaSq*spCor(d, phi)
                if(k == l):
                    C[int(l*m+k)] += tauSq

        ##########
        # dpotrf #
        ##########

        dpotrf(m, C)

        ##########
        # dpotri #
        ##########

        dpotri(m, C)

        #########
        # dysmv #
        #########

        dsymv(tmp_m, C, c, 0, 1, m, 0)

        # get the dot product of X and beta
                # dot_part = np.dot(X, p_beta_samples.T) # put outside
        dot_part2 = dot_part[:, s]
        dot_part3 = dot_part2[nn_indx_0[i, :]]


        for k in range(0, m, 1):
            d += tmp_m[k]*(Y_1D[nn_indx_0_1D[i + q*k]] - dot_part3[k])

        zIndx += 1


        # dot_part_Y = np.dot(X_0_1D, p_beta_samples.T) # put outside
        dot_part_Y2 = dot_part_Y[i, s]
                    # if(count == 1 or count == 2):
                    #     print("dot")
                    #     print((dot_part_Y2))

        # i = q (45), s = n_smaples (10)
        y_0[s*q+i] = dot_part_Y2 + d + np.sqrt(sigmaSq + tauSq - np.dot(tmp_m, c))*z[zIndx]



def rNNGPPredict(X, Y, coords, n, p, m, X_0, coords_0, q, nn_indx_0, 
                p_beta_samples, p_theta_samples, n_samples, verbose, progress_rep, n_reps):
    # print stuff off
    if(verbose):
        print("----------------------------------------------------------")
        print(" Prediction description")
        print("----------------------------------------------------------")
        print("NNGP Response model fit with " + str(n) + " observations.")
        print("Number of covariates " + str(p) + " (including intercept if specified).")
        print("Using the " + "exponential" + " spatial correlation model.")
        print("Using " + str(m) + " nearest neighbors.")
        print("Number of MCMC samples " + str(n_samples) + ".")
        print("Predicting at " + str(q) + " locations.")

    # indx
    nTheta = 3
    sigmaSqIndx = 0
    tauSqIndx = 1
    phiIndx = 2

    mm = m*m

    C = np.zeros(mm)
    c = np.zeros(m)
    tmp_m = np.zeros(m)
    phi = sigmaSq = tauSq = 0

    y_0 = np.zeros(q*n_samples)

    if(verbose):
        print("----------------------------------------------------------")
        print(" Predicting")
        print("----------------------------------------------------------")

    z = np.zeros(q*n_samples)
    zIndx = -1

    # reshape theta
    theta = np.array(np.reshape(p_theta_samples, -1, order='C'))

    # flatten both coords
    coords_1D =np.array(np.reshape(coords, -1, order='F'))
    coords_0_1D = np.array(np.reshape(coords_0, -1, order='F'))
    # flatten nnIndx0
    nn_indx_0_1D = np.array(np.reshape(nn_indx_0, -1, order='F'))

    X_0_1D = np.array(X_0)
    Y_1D = np.array(Y)

    z = np.random.normal(loc = 0, scale = 1, size = (q*n_samples))

    dot_part = np.dot(X, p_beta_samples.T)
    dot_part_Y = np.dot(X_0_1D, p_beta_samples.T)

    for i in range(0, q, 1):
        # used for timming
        if(progress_rep):
            if(i % n_reps == 0 or i == 0):
                start = time.time()
        # calculate y hat
        if(i == 1):
            starts = time.time()
        y_hat_calc(n_samples, theta, nTheta, phiIndx, sigmaSqIndx, tauSqIndx, coords_1D, m, nn_indx_0_1D, 
                   i, q, coords_0_1D, C, tmp_m, c, dot_part, dot_part_Y, nn_indx_0, Y_1D, zIndx, X_0_1D, n, y_0, z)
        if(i == 1):
            print("Sample Time:", time.time() - starts)

        # save samples
        if(progress_rep):
            if((i + 1) % n_reps == 0):                             # time for N samples * how many left/10
                print(100*((i+1))/(q), "%, Time estimate left = ", np.round((time.time() - start)*(q - (i+1))/n_reps, 2), "sec")

    if(verbose):
        print("Location: " + str(i + 1) + " of " + str(q) + ", " + str(100.0*(i + 1)/q))

    y_hat = np.array(y_0).flatten().reshape((q, n_samples), order = 'F')

    # print(pd.DataFrame(y_hat))

    return y_hat
