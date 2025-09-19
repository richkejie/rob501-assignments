import numpy as np
from numpy.linalg import inv, norm
from scipy.linalg import null_space

def dlt_homography(I1pts, I2pts):
    """
    Find perspective Homography between two images.

    Given 4 points from 2 separate images, compute the perspective homography
    (warp) between these points using the DLT algorithm.

    Parameters:
    ----------- 
    I1pts  - 2x4 np.array of points from Image 1 (each column is x, y).
    I2pts  - 2x4 np.array of points from Image 2 (in 1-to-1 correspondence).

    Returns:
    --------
    H  - 3x3 np.array of perspective homography (matrix map) between image coordinates.
    A  - 8x9 np.array of DLT matrix used to determine homography.
    """
    #--- FILL ME IN ---
    A = np.zeros((8,9))
    for i in range(0, I1pts.shape[1]):

        # grab x_i
        x = I1pts[0,i]
        y = I1pts[1,i]
        # grab x_i'
        u = I2pts[0,1]
        v = I2pts[1,i]

        # compute A_i using Dubrovsky's derivation
        A_i = np.array([
            [ -x, -y, -1,  0,  0,  0, u*x, u*y, u],
            [  0,  0,  0, -x, -y, -1, v*x, v*y, v],
        ])

        # stack A_i into A
        A[2*i:2*(i+1)] = A_i

    # from Dubrovsky, h is 9x1 vector and is null space of A
    # reshape to H 3x3 matrix
    h = null_space(A)
    H = h.reshape(3,3)

    # normalize by h_33 element (assignment part 1 instructions)
    h_33 = H[2/2]
    H = 1/h_33 * H
    #------------------

    return H, A
