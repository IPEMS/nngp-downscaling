###########################
# gets the beta and theta #
###########################

# for matrix
import numpy as np

# for GPU accel
from numba import njit

# files
from fortran_py import dpotrf, dpotri, dsymv, dgemv, daxpy
from util_py import logitInv, logit, mvrnorm, Q, dist2, spCor

# for test times
import time

# helper fucntions

@njit
def updateBF(B, F, c, C, coords, nnIndx, nnIndxLU, n, theta, tauSqIndx, sigmaSqIndx, phiIndx):
    logDet = 0

    for i in range(0, n, 1):
        if(i > 0):
            for k in range(0, int(nnIndxLU[int(n + i)]), 1):
                e = dist2(coords[int(i)], coords[int(n + i)], coords[int(nnIndx[int(nnIndxLU[int(i)]+k)])], coords[int(n+nnIndx[int(nnIndxLU[int(i)]+k)])])
                c[k] = theta[int(sigmaSqIndx)]*spCor(e, theta[phiIndx])

                for l in range(0, k + 1, 1):
                    e = dist2(coords[int(nnIndx[int(nnIndxLU[int(i)] + k)])], coords[int(n+nnIndx[int(nnIndxLU[int(i)]+k)])], coords[int(nnIndx[int(nnIndxLU[int(i)]+l)])], coords[int(n+nnIndx[int(nnIndxLU[int(i)]+l)])])
                    C[int(l*nnIndxLU[int(n+i)]+k)] = theta[sigmaSqIndx]*spCor(e, theta[phiIndx])
                    if(l == k):
                        C[int(l*nnIndxLU[int(n+i)]+k)] += theta[tauSqIndx]

            ##########
            # dpotrf #
            ##########

            dpotrf(int(nnIndxLU[int(n + i)]), C)

            ##########
            # dpotri #
            ##########

            dpotri(int(nnIndxLU[int(n + i)]), C)

            ###################################
            # Dsymv (y := alpha*A*x + beta*y) #
            ###################################
            
            dsymv(B, C, c, 0, 1, int(nnIndxLU[int(n + i)]), int(nnIndxLU[int(i)]))

            F[i] = theta[int(sigmaSqIndx)] - np.dot(B[nnIndxLU[i]:nnIndxLU[i]+nnIndxLU[n +i]],c[:nnIndxLU[n +i]]) + theta[int(tauSqIndx)]
        else:
            B[i] = 0
            F[i] = theta[sigmaSqIndx] + theta[tauSqIndx]

    logDet += np.sum(np.log(F))

    return logDet


# main function
def rNNGP(y, X, p, n, m, coords, cov_model, nn_indx, nn_indx_lu,
        sigma_sq_IG, tau_sq_IG, phi_Unif, beta_starting, sigma_sq_starting,
        tau_sq_starting, phi_starting, sigma_sq_tuning, tau_sq_tuning, phi_tuning, 
        n_samples, verbose, progress_rep, n_reps):
    

    # convert sIndx, coords, and nnIndxLU into arrays
    coords = (coords.T).flatten()
    nn_indx_lu = (nn_indx_lu.T).flatten()
    nn_indx = (nn_indx.T).flatten()

    # get args
        # already have them

    # priors
        # already have them

    # should have some thread stuff here
        # leave out for now

    if(verbose):
        print("----------------------------------------------------------")
        print(" Model description                                     ")
        print("----------------------------------------------------------")
        print("NNGP Response model fit with " + str(n) + " observations.")
        print("Number of covariates " + str(p) + " (including intercept if specified).")
        print("Using the " + str(cov_model) + " spatial correlation model.")
        print("Using " + str(m) + " nearest neighbors.")
        print("Number of MCMC samples " + str(n_samples) + ".\n")
        print("Priors and hyperpriors:")
        print("\tbeta flat.")
        print("\tsigma.sq IG hyperpriors shape=" + str(sigma_sq_IG[0]) + " and scale=" + str(sigma_sq_IG[1]))
        print("\ttau.sq IG hyperpriors shape="  + str(tau_sq_IG[0]) + " and scale="  + str(tau_sq_IG[1]))
        print("\tphi Unif hyperpriors a=" + str(phi_Unif[0]) + " and b=" + str(phi_Unif[1]))
        # if matern print some stuff

    # values for non matern style
    nTheta = 3
    sigma_sq_indx = 0
    tau_sq_indx = 1
    phi_indx = 2

# assuming here that both the beta and theta are being copied in.
    # starting
    beta = beta_starting
    theta = np.zeros(nTheta)

    # beta_starting = beta

    # set values
    theta = np.array([sigma_sq_starting, tau_sq_starting, phi_starting])

    # print("test theta")
    # print(theta)

    # tuning and fixed
    tuning = np.array([sigma_sq_tuning, tau_sq_tuning, phi_tuning])

    # other stuff
    nIndx = (1 + m)/2*m + (n - m - 1)*m
    mm = m*m
    thetaCand = np.zeros(nTheta)
    B = np.zeros(int(nIndx))
    F = np.zeros(n)
    c = np.zeros(m)
    C = np.zeros(mm)

    # need to return beta_samples, thetaSamples. and repSamples
    beta_samples = np.zeros((p, n_samples))
    theta_samples = np.zeros((nTheta, n_samples))

    # check for nRep
        # default is false

    # other stuff
    # double logPostCand, logPostCurrent, logDetCurrent, logDetCand, QCurrent, QCand;
    accept = batchAccept = status = 0
    pp = p*p
    tmp_pp = np.zeros((pp))
    temp_p = np.zeros(p)
    temp_p2 = np.zeros(p)
    temp_n = np.zeros(n)


    thetaUpdate = True

    logDetCurrent = updateBF(B, F, c, C, coords, nn_indx, nn_indx_lu, n, theta,
                            tau_sq_indx, sigma_sq_indx, phi_indx)

    #########
    # Dgemv #
    #########

    temp_n = dgemv(temp_n, 1, X, beta, 0)

                # print("---temp_n 1---")
                # print(temp_n)

    #########
    # Daxpy #
    #########

    # convert y to np
    y = np.array(y)

    temp_n = daxpy(temp_n, -1, y)

                # print("---temp_n 2---")
                # print(temp_n)


    QCurrent = Q(B, F, temp_n, temp_n, n, nn_indx, nn_indx_lu)


    if(verbose):
        print("----------------------------------------------------------")
        print(" Sampling                                              ")
        print("----------------------------------------------------------")

    # make X flat
    X_flat = (X.T).flatten()


    ###############
    # Update Beta #
    ###############

    for s in range(0, n_samples, 1):
    # call update beta theta
        # used for timming
        if(progress_rep):
            if(s %  n_reps == 0 or s == 0):
                start = time.time()
        if(thetaUpdate):
            thetaUpdate = False

            ###############
            # Update Beta #
            ###############

            for i in range(0, p, 1):
                temp_p[i] = Q(B, F, X_flat[(n*i):], y, n, nn_indx, nn_indx_lu)
                for j in range(0, i + 1, 1):
                    tmp_pp[j*p + i] = Q(B, F, X_flat[(n*j):], X_flat[(n*i):], n, nn_indx, nn_indx_lu)

            ##########
            # dpotrf #
            ##########

            dpotrf(p, tmp_pp)

            ##########
            # dpotri #
            ##########

            dpotri(p, tmp_pp)

            #########
            # dsymv #
            #########

            dsymv(temp_p2, tmp_pp, temp_p, 0, 1, p, 0)

            ##########
            # dpotrf #
            ##########
            
            dpotrf(p, tmp_pp)



        beta = mvrnorm(beta, temp_p2, tmp_pp, p)

                    # ### setting beta
                    # beta = np.array([0.1302498919,	0.8452406535,	0.9995658252])
                    # print("beta")
                    # print(beta)

        ################
        # Update Theta #
        ################


        #########
        # Dgemv #
        #########

        # dgemv
            # dgemv -> y = alpha*A*x + beta*y
                # alpha -> 1, A -> X, x -> beta, Y -> temp_n
                # (dgemv)(ntran, &n, &p, &one, X, &n, beta, &inc, &zero, tmp_n, &inc FCONE);
        temp_n = dgemv(temp_n, 1, X, beta, 0)

                    # print("---temp_n 3---")
                    # print(temp_n)

        #########
        # Daxpy #
        #########

        # daxpy
            # (daxpy)(&n, &negOne, y, &inc, tmp_n, &inc);
                # daxpy <- dy = dy + da*dx
                # temp_n = temp_n - y
        temp_n = daxpy(temp_n, -1, y)
    
                    # print("---temp_n 4---")
                    # print(temp_n)
        
        if(progress_rep):
            if(s == 0):
                start = time.time()

        logDetCurrent = updateBF(B, F, c, C, coords, nn_indx, nn_indx_lu, n, theta,           
                                                    tau_sq_indx, sigma_sq_indx, phi_indx)
        if(progress_rep):
            if(s == 0):
                print("logDetCurrent Time = " + str((time.time() - start)) + " sec")
        # print(logDetCurrent)

        if(progress_rep):
            if(s == 0):
                start = time.time()

        QCurrent = Q(B, F, temp_n, temp_n, n, nn_indx, nn_indx_lu)

        if(progress_rep):
            if(s == 0):
                print("QCurrent Time = " + str((time.time() - start)) + " sec")

        logPostCurrent = -0.5*logDetCurrent - 0.5*QCurrent
        logPostCurrent += np.log(theta[phi_indx] - phi_Unif[0]) + np.log(phi_Unif[1] - theta[phi_indx])
        logPostCurrent += -1*(1 + sigma_sq_IG[0])*np.log(theta[sigma_sq_indx]) - sigma_sq_IG[1]/theta[sigma_sq_indx] + np.log(theta[sigma_sq_indx])
        logPostCurrent += -1*(1 + tau_sq_IG[0])*np.log(theta[tau_sq_indx]) - tau_sq_IG[1]/theta[tau_sq_indx] + np.log(theta[tau_sq_indx])

        # candiate
        thetaCand[phi_indx] = logitInv(np.random.normal(loc = logit(theta[phi_indx], phi_Unif[0], phi_Unif[1]), scale = tuning[phi_indx]), phi_Unif[0], phi_Unif[1])
        thetaCand[sigma_sq_indx] = np.exp(np.random.normal(loc = np.log(theta[sigma_sq_indx]), scale = tuning[sigma_sq_indx]))
        thetaCand[tau_sq_indx] = np.exp(np.random.normal(loc = np.log(theta[tau_sq_indx]), scale = tuning[tau_sq_indx]))

        logDetCand = updateBF(B, F, c, C, coords, nn_indx, nn_indx_lu, n, thetaCand,
                                                    tau_sq_indx, sigma_sq_indx, phi_indx)

        QCand = Q(B, F, temp_n, temp_n, n, nn_indx, nn_indx_lu)

        logPostCand = -0.5*logDetCand - 0.5*QCand
        logPostCand += np.log(thetaCand[phi_indx] - phi_Unif[0]) + np.log(phi_Unif[1] - thetaCand[phi_indx])
        logPostCand += -1*(1 + sigma_sq_IG[0])*np.log(thetaCand[sigma_sq_indx]) - sigma_sq_IG[1]/thetaCand[sigma_sq_indx] + np.log(thetaCand[sigma_sq_indx])
        logPostCand += -1*(1 + tau_sq_IG[0])*np.log(thetaCand[tau_sq_indx]) - tau_sq_IG[1]/thetaCand[tau_sq_indx] + np.log(thetaCand[tau_sq_indx])

        if(np.random.uniform(low = 0.0, high = 1.0) <= np.exp(logPostCand - logPostCurrent)):
        # if(np.random.uniform(a = 0.0, b = 1.0) <= np.exp(logPostCand - logPostCurrent)):
            thetaUpdate = True

            #################
            # Fix for Theta #
            #################
            
            for l in range(0, nTheta, 1):
                theta[l] = thetaCand[l]
            accept += 1
            batchAccept += 1

        beta_samples[:, s] = beta
        theta_samples[:, s] = theta

        # save samples
        if(progress_rep):
            if((s + 1) % n_reps == 0):                                      # time for N samples * how many left/10
                print(100*(s+1)/n_samples, "%, Time estimate left = ", np.round((time.time() - start)*(n_samples - (s+1))/n_reps, 2), "sec")

        status += 1

    if(verbose):
        print("----------------------------------------------------------")
        print("Sampled: " + str(s + 1) + " of " + str(n_samples) + ", " + str(100*((s+ 1)/n_samples)) + "%")
        print("Report interval Metrop. Acceptance rate: " + str(round((100*(batchAccept/n_samples)), 4)) + "%")
        print("Overall Metrop. Acceptance rate: " + str(round((100*(accept/n_samples)), 4)) + "%")
        print("----------------------------------------------------------")

    return beta_samples, theta_samples
