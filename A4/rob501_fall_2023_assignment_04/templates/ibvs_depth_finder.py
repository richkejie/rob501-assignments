import numpy as np
from numpy.linalg import inv
from ibvs_jacobian import ibvs_jacobian

def ibvs_depth_finder(K, pts_obs, pts_prev, v_cam):
    """
    Compute estimated 

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K        - 3x3 np.array, camera intrinsic calibration matrix.
    pts_obs  - 2xn np.array, observed (current) image plane points.
    pts_prev - 2xn np.array, observed (previous) image plane points.
    v_cam    - 6x1 np.array, camera velocity (last commmanded).

    Returns:
    --------
    zs_est - nx0 np.array, updated, estimated depth values for each point.
    """
    n = pts_obs.shape[1]
    J = np.zeros((2*n, 6))
    zs_est = np.zeros(n)

    #--- FILL ME IN ---
    v = v_cam[0:3]
    w = v_cam[3:6]

    # for each point, compute Jacobian and find estimated depth
    for i in range(n):
        # use depth of 1 as estimate for now
        J = ibvs_jacobian(K,pts_obs[:,i].reshape(-1,1),z=1)
        # get J_t and J_w from J
        J_t = J[:,0:3]
        J_w = J[:,3:6]
        # solve for depth using equations from Corke text
        A = J_t @ v
        pts_vel = (pts_obs[:,i] - pts_prev[:,i]).reshape(-1,1)
        b = pts_vel - J_w @ w
        # Ax = b, x = 1/z, solve for x using least squares
        x = inv(A.T @ A) @ A.T @ b
        zs_est[i] = 1/x

    #------------------

    correct = isinstance(zs_est, np.ndarray) and \
        zs_est.dtype == np.float64 and zs_est.shape == (n,)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return zs_est