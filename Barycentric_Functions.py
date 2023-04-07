import numpy as np

def bsplinen_bary2cart(simplex, lambd):
    """
    Convert barycentric coordinates to Cartesian coordinates
    
    Parameters
    ----------
    simplex : numpy.ndarray
        Vertices of the simplex
    lambd : numpy.ndarray
        Barycentric coordinates
    
    Returns
    -------
    X : numpy.ndarray
        Cartesian coordinates
    """
    Lcount = lambd.shape[0]
    X = np.zeros((Lcount, lambd.shape[1] - 1))
    
    v0 = simplex[0, :]
    vcount2 = simplex.shape[0]
    
    for j in range(Lcount):
        for i in range(1, vcount2):
            X[j, :] = X[j, :] + lambd[j, i] * (simplex[i, :] - v0)
        X[j, :] = X[j, :] + v0
    
    return X


def bsplinen_cart2bary(simplex, X):
    """
    Convert Cartesian coordinates to barycentric coordinates
    
    Parameters
    ----------
    simplex : numpy.ndarray
        Vertices of the simplex
    X : numpy.ndarray
        Cartesian coordinates
    
    Returns
    -------
    Lambda : numpy.ndarray
        Barycentric coordinates
    """
    # The reference vertex is always chosen as the first simplex vertex.
    # This can be done because barycentric coordinates are not dependent on
    # the reference point.
    v0 = simplex[0, :]
    vcount2 = simplex.shape[0] - 1
    Xcount = X.shape[0]
    
    Lambda = np.zeros((Xcount, vcount2 + 1))
    #vcount2 = length(simplex(:, 1)) - 1;
    
    # assemble matrix A
    A = np.zeros((vcount2, vcount2))
    count = 1
    for i in range(1, simplex.shape[0]):
        A[:, count] = (simplex[i, :] - v0)
        count = count + 1
    
    
    for i in range(Xcount):
        # relative coordinates of x
        p = (X[i, :] - v0)

        # the last (n) barycentric coordinates. 
        lambda1 = np.linalg.solve(A, p)

        # the first barycentric coordinate; lambda0
        lambda0 = 1 - sum(lambda1)

        # insert lambda0 into the Lambda vector
        Lambda[i, 0] = lambda0
        Lambda[i, 1:] = lambda1
    
    return Lambda
