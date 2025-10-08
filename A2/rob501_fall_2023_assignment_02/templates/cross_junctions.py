import numpy as np
from numpy.linalg import inv, lstsq
from scipy.linalg import null_space
from scipy.ndimage.filters import *
from matplotlib.path import Path

# You may add support functions here, if desired.

# part 1
def saddle_point(I):
    M, N = I.shape
    A = np.empty((M*N, 6))
    b = np.empty((M*N,1))
    i = 0
    for y in range(0,M):
        for x in range(0,N):
            A[i,:] = [x*x, x*y, y*y, x, y, 1]
            b[i] = I[y,x]
            i += 1
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
    pt = intersection(ALPHA, BETA, GAMMA, DELTA, EPSILON, ZETA)

    correct = isinstance(pt, np.ndarray) and \
        pt.dtype == np.float64 and pt.shape == (2, 1)
    if not correct:
        raise TypeError("Wrong type or size returned!")
    return pt

# from A1
def dlt_homography(I1pts, I2pts):
    num_points = I1pts.shape[1]
    A = np.zeros((8,9))
    for i in range(0, num_points):
        x = I1pts[0,i]
        y = I1pts[1,i]
        u = I2pts[0,i]
        v = I2pts[1,i]
        A_i = np.array([
            [ -x, -y, -1,  0,  0,  0, u*x, u*y, u],
            [  0,  0,  0, -x, -y, -1, v*x, v*y, v],
        ])
        A[2*i:2*(i+1)] = A_i
    h = null_space(A)
    H = h.reshape(3,3)
    h_33 = H[2,2]
    H = 1/h_33 * H
    return H, A

def cross_junctions(I, bpoly, Wpts):
    """
    Find cross-junctions in image with subpixel accuracy.

    The function locates a series of cross-junction points on a planar 
    calibration target, where the target is bounded in the image by the 
    specified quadrilateral. The number of cross-junctions identified 
    should be equal to the number of world points.

    Note also that the world and image points must be in *correspondence*,
    that is, the first world point should map to the first image point, etc.

    Parameters:
    -----------
    I      - Single-band (greyscale) image as np.array (e.g., uint8, float).
    bpoly  - 2x4 np.array, bounding polygon (clockwise from upper left).
    Wpts   - 3xn np.array of world points (in 3D, on calibration target).

    Returns:
    --------
    Ipts  - 2xn np.array of cross-junctions (x, y), relative to the upper
            left corner of the target. The array must contain float64 values.
    """
    #--- FILL ME IN ---

    # Code goes here...

    N = Wpts.shape[1] # number of world points
    Ipts = np.empty((2,N)) # will hold cross-junction points
    
    # estimate world bounding box of checkerboard
    W_xmin = min(Wpts[0,:])
    W_xmax = max(Wpts[0,:])
    W_ymin = min(Wpts[1,:])
    W_ymax = max(Wpts[1,:])

    def box_side_length_est(Wpts):
        return Wpts[0,1] - Wpts[0,0] # just use size of first gridbox for estimation (in x-direction)

    box_side_length = box_side_length_est(Wpts)

    # update mins/maxs to get estimate for edge of checkerboard
    # edge of squares would be 1*box_side_length away
    # edge of board in x-dir would be approx another 1/2*box_side_length away
    # edge of board in y-dir would be approx another 1/4*box_side_length away (smaller y-border)
    W_xmin -= (1+1/2) * box_side_length
    W_xmax += (1+1/2) * box_side_length
    W_ymin -= (1+1/4) * box_side_length
    W_ymax += (1+1/4) * box_side_length

    # assemble bounding box
    W_bbox = np.array([
        [W_xmin, W_xmax, W_xmax, W_xmin],
        [W_ymin, W_ymin, W_ymax, W_ymax],
    ])

    # apply DLT to get homography
    H, A = dlt_homography(W_bbox, bpoly)

    # get initial junction point estimates from homography
    Wpts[2] = 1 # assignment instrs say can assume z = 1 for all world points
    I_junctions = np.matmul(H,Wpts)
    I_junctions /= I_junctions[2] # normalize by z-element
    I_junctions = np.round(I_junctions[:2]).astype(int) # round to int locations, get rid of z-element
                                                        # 2xn (2 rows for x,y coords; n columns for n points)
                                                        
    # search in region around estimate for saddle
    # for better estimate of junction point
    space = 15 # check 15 pxs around initial estimate
    for i in range(N):
        xmin = I_junctions[0,i] - space
        xmax = I_junctions[0,i] + space
        ymin = I_junctions[1,i] - space
        ymax = I_junctions[1,i] + space

        region = I[ymin:ymax+1,xmin:xmax+1]
        junction_saddle_est = saddle_point(region).flatten() + I_junctions[:,i] - space
        Ipts[:,i] = junction_saddle_est

    #------------------

    correct = isinstance(Ipts, np.ndarray) and \
        Ipts.dtype == np.float64 and Ipts.shape[0] == 2

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Ipts