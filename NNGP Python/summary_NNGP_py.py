#################################
# gets the summary of the model #
#################################

import pandas as pd
import numpy as np

# for printing the theta/beta
    # plotting
import matplotlib.pyplot as plt

def summary(object):

    X = object[8]
    p = len(X[0])

    p_theta_samples = object[5]
    p_beta_samples = object[6]
    n_samples = len(p_theta_samples)

    print("------------------------------------------------------")
    print(" Summary of NNGP")
    print("------------------------------------------------------")

    
    ###############################
    # test what is if model works #
    ###############################

    pd.set_option('display.max_columns', 10)


    # transpose
    p_beta_samplesT = p_beta_samples.T
    p_theta_samplesT = p_theta_samples.T

    percentile = np.array([2.5, 25, 50, 75, 97.5])

    # holds all constants in front of each x
    percentile_beta = np.zeros((p, len(percentile)))
    # holds sigam.sq, tau.sq, and phi
    percentile_theta = np.zeros((3, len(percentile)))

    print("Model class is NNGP, method response, family gaussian.")
    print("Model object contains 10 MCMC samples.")
    print("Chain sub.sample:")
    print("start = " + str(int(n_samples/2)))
    print("end = " + str(n_samples))
    print("thin = 1")
    print("Sample Size = " + str(n_samples - int(n_samples/2) + 1))

    for i in range(0, len(percentile), 1):
        for j in range(0, p, 1):
            percentile_beta[j, i] = np.percentile(p_beta_samplesT[j, :int(n_samples/2)], percentile[i])
        for k in range(0, 3, 1):
            percentile_theta[k, i] = np.percentile(p_theta_samplesT[k, :int(n_samples/2)], percentile[i])


    all = pd.DataFrame(np.concatenate((percentile_beta, percentile_theta), axis = 0))

    all.columns = ['2.5%', '25%', '50%', '75%', '97.5%']

    if(p == 5):
        all.index = ['(Intercept)', 'X2', 'X3', 'X4', 'X5', 'sigma_sq', 'tau_sq', 'phi']
    if(p == 3):
        all.index = ['(Intercept)', 'X2', 'X3', 'sigma_sq', 'tau_sq', 'phi']

    print(pd.DataFrame(np.around(all, 2)))

    ############
    # plotting #
    ############
    x_vals = np.arange(0, n_samples, 1)

    if(p == 5):
        for i in range(0, 5, 1):
            plt.subplot(2, 3, i+1)
            plt.plot(x_vals, p_beta_samplesT[i, :])
            plt.title(all.index[i])
        plt.show()
        for i in range(5, 8, 1):
            plt.subplot(2, 2, i+1-5)
            plt.plot(x_vals, p_theta_samplesT[i-5, :])
            plt.title(all.index[i])
        plt.show()
    if(p == 3):
        for i in range(0, 3, 1):
            plt.subplot(2, 2, i+1)
            plt.plot(x_vals, p_beta_samplesT[i, :])
            plt.title(all.index[i])
        plt.show()
        for i in range(3, 6, 1):
            plt.subplot(2, 2, i+1-3)
            plt.plot(x_vals, p_theta_samplesT[i-3, :])
            plt.title(all.index[i])
        plt.show()
