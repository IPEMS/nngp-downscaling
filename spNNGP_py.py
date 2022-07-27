# main file for spNNGP

# get the rNNGP function
from rNNGP_py import rNNGP

# get nn
from nn_py import mkNNIndexCB_in_nn

# used for matrix
import numpy as np

# used for linear model
from sklearn.linear_model import LinearRegression

# helper function
def mkNNIndexCB(coords, m, n_omp_threads = 1):
    n = len(coords)
    nIndex = (1 + m)/2*m + (n - m - 1)*m
    nnIndex = np.repeat(0.0, nIndex)
    nnDist = np.repeat(0.0, nIndex)
    nnIndexLU = np.zeros((n, 2))

    # need to call the next mkNNIndexCB function. Originally in C++
        # call function
    nnIndex, nnDist, nnIndexLU = mkNNIndexCB_in_nn(n, m, coords, nnIndex, nnDist, nnIndexLU, n_omp_threads)
    # calcualte time

    return nnIndex, nnDist, nnIndexLU

def spNNGP(x, y, coords, starting, tuning, priors, n_samples,
        method = "response", family = "gaussian", n_neighbors = 15, 
        cov_model = "exponential", verbose = True, n_report = 100):

    ###########
    # Formula #
    ###########
    # y ~ x -1

    # column
    p = len(x[0])
    # rows
    n = len(x)

    # can check that coords is a n x 2 matrix
    if(len(coords) != n or  len(coords[0]) != 2):
        print("Error coords is wrong! Needs to be n x 2")
    
    ##########
    # Family #
    ##########

    # since family should always be gaussian...
        # weights are not set (ony for binomial)

    ##########################
    # Neighbors and ordering #
    ##########################

    # ordering the data
    ord = np.argsort(coords[:, 0]) # order in terms of x

    # save old
    coords_old = coords
    x_old = x
    y_old = y

    # get the orderd data points
    coords = coords[ord]
    x = x[ord]
    y = y[ord]

    ###############
    # NNGP Method #
    ###############

    ####################
    # Covariance Model #
    ####################

    # using the exponential cov model

    ##########
    # Priors #
    ##########

    # default values
    sigma_sq_IG = 0
    tau_sq_IG = 0
    phi_Unif = 0

    # actual values
    sigma_sq_IG = priors[1] # where sigma_sq_IG is stored
    tau_sq_IG = priors[2] # where tau_sq_IG is stored
    phi_Unif = priors[0] # where phi_Unif_IG is stored

    ###################
    # Starting Values #
    ###################

    # initual values of 0
    beta_starting = 0
    sigma_sq_starting = 0
    tau_sq_starting = 0
    phi_starting = 0
    
    ####################
    # get coefficients #
    ####################

    # set up the equation
    reg = LinearRegression().fit(x, y)
    # coefficients in a matrix
    
    beta_starting = np.flip(np.append(reg.coef_[1], reg.intercept_))
    
    # get each beta
    for i in range(2, p, 1):
        beta_starting = np.append(beta_starting, reg.coef_[i])

    # get the other starting values
    sigma_sq_starting = starting[1]
    tau_sq_starting = starting[2]
    phi_starting = starting[0]

    #################
    # Tuning Values #
    #################
    sigma_sq_tuning = 0
    tau_sq_tuning = 0
    phi_tuning = 0

    # get the actual values
    sigma_sq_tuning = tuning[1]
    tau_sq_tuning = tuning[2]
    phi_tuning = tuning[0] 

    if(verbose == True): # by default
        print("----------------------------------------------------------")
        print(" Building the neighbor list                            ")
        print("----------------------------------------------------------")

    # cb search type
        # also can use run_time, but I think not needed
                # print(self.coords)
                # print(self.n_neighbors)
                # print(self.n_omp_threads)

    nn_index, nnDist, nn_index_lu = mkNNIndexCB(coords, n_neighbors)
   
    ###########################
    # fitted and replicated y #
    ###########################

    # fit_rep is faulse by default, so it has been ommited

    # fit_rep is false so skip

    cov_model_indx = 0 # for exponential (example)

    ##############################
    # pack it up and off it goes #
    ##############################


                    # print("---Y---")
                    # print(y)
                    # print("---X---")
                    # print(x)
                    # print("---p---")
                    # print(p)
                    # print("---n---")
                    # print(n)
                    # print("---n_neighbors---")
                    # print(n_neighbors)
                    # print("---coords---")
                    # print(coords)
                    # print("---cov_model---")
                    # print(cov_model)
                    # print("---nn_index---")
                    # print(nn_index)
                    # print("---nn_index_lu---")
                    # print(nn_index_lu)
                    # print("---sigma_sq_IG---")
                    # print(sigma_sq_IG)
                    # print("---tau_sq_IG---")
                    # print(tau_sq_IG)
                    # print("---phi_Unif---")
                    # print(phi_Unif)
                    # print("---beta_starting---")
                    # print(beta_starting)
                    # print("---sigma_sq_starting---")
                    # print(sigma_sq_starting)
                    # print("---tau_sq_starting---")
                    # print(tau_sq_starting)
                    # print("---phi_starting---")
                    # print(phi_starting)
                    # print("---sigma_sq_tuning---")
                    # print(sigma_sq_tuning)
                    # print("---tau_sq_tuning---")
                    # print(tau_sq_tuning)
                    # print("---phi_tuning---")
                    # print(phi_tuning)
                    # print("---n_samples---")
                    # print(n_samples)
                    # print("---verbose---")
                    # print(verbose)
    



    if(family == "gaussian"):
        if(method == "response"):
            # call rNNGP

            beta_out, theta_out = rNNGP(y, x, p, n, n_neighbors, coords, cov_model, nn_index, nn_index_lu,
                        sigma_sq_IG, tau_sq_IG, phi_Unif, beta_starting, sigma_sq_starting,
                        tau_sq_starting, phi_starting, sigma_sq_tuning, tau_sq_tuning, phi_tuning,
                        n_samples, verbose)

        else:
            # call sNNGP
            print("do the sequential")

    beta_s = beta_out.T
    theta_s = theta_out.T


    #################
    # Return values #
    #################

    # put back in the original order
    coords = coords_old
    x = x_old
    y = y_old

    if(method == "response"):
            #  0       1               2           3        4        5        6    7   8  9     10             11
        return n, n_neighbors, nn_index, nn_index_lu, ord, theta_s, beta_s, coords, x, y, family, cov_model_indx
            # can also return runtimes if used
    # else:
        # return n, n_neighbors, nn_index, nn_index_lu, u_index, u_index_lu, ui_index, ord,
            # can also return runtimes if used