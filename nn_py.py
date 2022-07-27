################################ 
# used for different functions #
################################

# matrix
import numpy as np

# GPU accel
from numba import njit

@njit
def dmi(x, c, inc):
    return (x[0]+x[inc]-c[0]-c[inc])**2

@njit
def dei(x, c, inc):
    return (x[0]-c[0])**2+(x[inc]-c[inc])**2

@njit
def getNNIndx(i, m):
    if(i == 0):
        iNNIndx = 0
        iNN = 0
    elif(i < m):
        iNNIndx = float((i)/2*(i-1))
        iNN = i
    else:
        iNNIndx = float((m)/2*(m-1)+(i-m)*m)
        iNN = m
    return iNNIndx, iNN

@njit
def fastNN(m, n, coords, ui, u, sIndx, rSIndx, rSNNDist):

    # ui = is the index for which we need the m nearest neighbors
    # m = number of nearest neighbors
    # n = number of observations, i.e., length of u
    # sIndx = the NNGP ordering index of length n that is pre-sorted by u
    # u = x+y vector of coordinates assumed sorted on input
    # rSIndx = vector or pointer to a vector to store the resulting nn sIndx (this is at most length m for ui >= m)
    # rNNDist = vector or point to a vector to store the resulting nn Euclidean distance (this is at most length m for ui >= m) 

    #  'rsort_with_index' sorts on X, and applies the same permutation to
    #  INDEX.  'NA's are sorted last.
    # print("---------- fastNN ----------")


    #rSNNDist will hold de (i.e., squared Euclidean distance) initially.
    for i in range(0, m, 1):
        rSNNDist[i] = np.inf
        # print(rSNNDist[i])

    up = down = True
    i = j = ui

    # print("---------- tests ----------")

    while(up or down):
        if(i == 0):
            down = False
        if(j == (n-1)):
            up = False
        if(down):
            i = i - 1
            dm  = dmi(coords[int(sIndx[ui]):], coords[int(sIndx[i]):], n)
            if(dm > 2*rSNNDist[m -1]):
                down = False
            else:
                de = dei(coords[int(sIndx[ui]):], coords[int(sIndx[i]):], n)
                # print("de" + str(de))
                if(de < rSNNDist[m - 1] and sIndx[i] < sIndx[ui]):
                    rSNNDist[m-1] = de
                    rSIndx[m-1] = sIndx[i]
                    # sort the indx
                    # print("test1")
                    
                    test = np.argsort(rSNNDist[:m])
                    rSNNDist = rSNNDist[test]
                    rSIndx = rSIndx[test]

        if(up):
            j = j + 1
            dm = dmi(coords[int(sIndx[ui]):], coords[int(sIndx[j]):], n)
            # print("dm " + str(dm))
            if(dm > 2*rSNNDist[m-1]):
                up = False
            else:
                de = dei(coords[int(sIndx[ui]):], coords[int(sIndx[j]):], n)
                if(de < rSNNDist[m - 1] and sIndx[j] < sIndx[ui]):
                    rSNNDist[(m-1)] = de
                    rSIndx[(m-1)] = sIndx[j]

                    # print(de)
                    # print(sIndx[j])
                    
                    # sort the indx
                    # print("test2")
                    test = np.argsort(rSNNDist[:m])
                    rSNNDist = rSNNDist[test]
                    rSIndx = rSIndx[test]

                    # print(rSNNDist)
                    # print(rSIndx)
                    

    for i in range(0, m, 1):
        rSNNDist[i] = np.sqrt(rSNNDist[i])

    # print("------ test ------")
    # print(rSNNDist)
    # print(rSIndx)
    return rSIndx, rSNNDist

# @njit
def mkNNIndexCB_in_nn(n, m, coords, nnIndex, nnDist, nnIndexLU, nThreads):
    # i, innIndx, iNN
    # threads = 1

    # initialize arrays
    sIndx = np.zeros((n))
    u = np.zeros((n))

    u = coords[:, 0]+ coords[:, 1]

    # create the sorted index based on x + y 
    sIndx = np.argsort((u))

    # convert sIndx, coords, and nnIndxLU into arrays
    coords = (coords.T).flatten()
    nnIndexLU = (nnIndexLU.T).flatten()

    for i in range(0, n, 1):
        iNNIndx, iNN = getNNIndx(sIndx[i], m) # passes also iNNIndx, and iNN
        nnIndexLU[sIndx[i]] = iNNIndx
        nnIndexLU[n + sIndx[i]] = iNN

                    # # c++ &array[spot] -> py array[spot:]
                    # # nnIndex[int(iNNIndx):], nnDist[int(iNNIndx):] =
                    # print("iNN")
                    # print(iNN)
                    # print("n")
                    # print(n)
        Indx_part, Dist_part = fastNN(iNN, n, coords, i, u, sIndx, nnIndex[int(iNNIndx):], nnDist[int(iNNIndx):])

        # print("nnDist")
        for i in range(0, iNN, 1):
            nnDist[int(i+iNNIndx)] = Dist_part[int(i)]
            nnIndex[int(i+iNNIndx)] = Indx_part[int(i)]

        # print(nnIndex)
        # print(nnDist)

    # convert back to n x 2
    coords = np.column_stack((coords[:n], coords[n:]))
    nnIndexLU = np.column_stack((nnIndexLU[:n], nnIndexLU[n:]))
    
    return nnIndex, nnDist, nnIndexLU 