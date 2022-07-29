####################
# Setup Prediction #
####################

# files
        # from rNNGP_Predict_py import rNNGPPredict
from rNNGPPredict_py import rNNGPPredict

# for nn search
from scipy import spatial

def predict(object, X_0, coords_0, verbose = True, progress_rep = True, n_reps = 10):
    
    X = object[8]
    Y = object[9]
    coords = object[7]
    
    # column
    p = len(X[0])
    # rows
    n = len(X)

    p_theta_samples = object[5]
    p_beta_samples = object[6]
    n_samples = len(p_theta_samples)

    n_neighbors = object[1]
    cov_model_indx = object[11]

    # do a bunch of checks of X.0 and coords.0
        # riws
    q = len(X_0)

    tree = spatial.KDTree(coords)

    dists, nn_indx_0 = tree.query(coords_0, n_neighbors)


    y_return = rNNGPPredict(X, Y, coords, n, p, n_neighbors, X_0, coords_0, q, nn_indx_0, 
                        p_beta_samples, p_theta_samples, n_samples, verbose, progress_rep, n_reps)




    return y_return
