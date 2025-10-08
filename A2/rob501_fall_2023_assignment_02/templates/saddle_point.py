import numpy as np
from numpy.linalg import inv, lstsq

def saddle_point(I):
    """
    Locate saddle point in an image patch.

    The function identifies the subpixel centre of a cross-junction in the
    image patch I, by fitting a hyperbolic paraboloid to the patch, and then 
    finding the critical point of that paraboloid.

    Note that the location of 'pt' is relative to (-0.5, -0.5) at the upper
    left corner of the patch, i.e., the pixels are treated as covering an 
    area of one unit square.

    Parameters:
    -----------
    I  - Single-band (greyscale) image patch as np.array (e.g., uint8, float).

    Returns:
    --------
    pt  - 2x1 np.array (float64), subpixel location of saddle point in I (x, y).
    """
    #--- FILL ME IN ---
 
    # Code goes here.

    # solve for ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA
    # from (4) of Lucchese using linear least squares
    # then find intersection of lines using matrix formulation
    # also from Lucchese

    # dimensions
    M, N = I.shape

    # coefficients matrix
    A = np.empty((M*N, 6))
    # target Intensity values vector
    b = np.empty((M*N,1))

    # setup A and b according to (4) for all points in I
    i = 0
    for y in range(0,M):
        for x in range(0,N):
            A[i,:] = [x*x, x*y, y*y, x, y, 1]
            b[i] = I[y,x]
            i += 1

    # solve lstsq for parameters
    ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA = lstsq(A, b, rcond=None)[0].T[0]

    def intersection(ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA):

        M = np.array([
            [ 2*ALPHA,   BETA    ],
            [ BETA,      2*GAMMA ],
        ])
        x = np.array([
            [DELTA],
            [EPSILON],
        ])
        pt = -np.matmul(inv(M),x)

        return pt
    
    # find intersection
    pt = intersection(ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA)

    #------------------

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return pt