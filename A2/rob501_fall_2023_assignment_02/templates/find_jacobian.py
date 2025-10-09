import numpy as np
from numpy.linalg import inv

def dcm_from_rpy(rpy):
    """
    Rotation matrix from roll, pitch, yaw Euler angles.

    The function produces a 3x3 orthonormal rotation matrix R
    from the vector rpy containing roll angle r, pitch angle p, and yaw angle
    y.  All angles are specified in radians.  We use the aerospace convention
    here (see descriptions below).  Note that roll, pitch and yaw angles are
    also often denoted by phi, theta, and psi (respectively).

    The angles are applied in the following order:

     1.  Yaw   -> by angle 'y' in the local (body-attached) frame.
     2.  Pitch -> by angle 'p' in the local frame.
     3.  Roll  -> by angle 'r' in the local frame.  

    Note that this is exactly equivalent to the following fixed-axis
    sequence:

     1.  Roll  -> by angle 'r' in the fixed frame.
     2.  Pitch -> by angle 'p' in the fixed frame.
     3.  Yaw   -> by angle 'y' in the fixed frame.

    Parameters:
    -----------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.

    Returns:
    --------
    R  - 3x3 np.array, orthonormal rotation matrix.
    """
    cr = np.cos(rpy[0]).item()
    sr = np.sin(rpy[0]).item()
    cp = np.cos(rpy[1]).item()
    sp = np.sin(rpy[1]).item()
    cy = np.cos(rpy[2]).item()
    sy = np.sin(rpy[2]).item()

    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [  -sp,            cp*sr,            cp*cr]])

def rpy_from_dcm(R):
    """
    Roll, pitch, yaw Euler angles from rotation matrix.

    The function computes roll, pitch and yaw angles from the
    rotation matrix R. The pitch angle p is constrained to the range
    (-pi/2, pi/2].  The returned angles are in radians.

    Inputs:
    -------
    R  - 3x3 orthonormal rotation matrix.

    Returns:
    --------
    rpy  - 3x1 np.array of roll, pitch, yaw Euler angles.
    """
    rpy = np.zeros((3, 1))

    # Roll.
    rpy[0] = np.arctan2(R[2, 1], R[2, 2])

    # Pitch.
    sp = -R[2, 0]
    cp = np.sqrt(R[0, 0]*R[0, 0] + R[1, 0]*R[1, 0])

    if np.abs(cp) > 1e-15:
      rpy[1] = np.arctan2(sp, cp)
    else:
      # Gimbal lock...
      rpy[1] = np.pi/2
  
      if sp < 0:
        rpy[1] = -rpy[1]

    # Yaw.
    rpy[2] = np.arctan2(R[1, 0], R[0, 0])

    return rpy

def find_jacobian(K, Twc, Wpt):
    """
    Determine the Jacobian for NLS camera pose optimization.

    The function computes the Jacobian of an image plane point with respect
    to the current camera pose estimate, given a landmark point. The 
    projection model is the simple pinhole model.

    Note that the homogeneous transformation matrix provided defines the
    transformation from the *camera frame* to the *world frame* (to 
    project into the image, you would need to invert this matrix).

    Parameters:
    -----------
    K    - 3x3 np.array, camera intrinsic calibration matrix.
    Twc  - 4x4 np.array, homogenous pose matrix, current guess for camera pose. 
    Wpt  - 3x1 world point on calibration target (one of n).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian matrix (columns are tx, ty, tz, r, p, q).
         The Jacobian must contain float64 values.
    """
    #--- FILL ME IN ---
 
    # Code goes here...
    
    # get rotation matrix and translation vector from Twc
    Cwc = Twc[:3,:3]
    t = Twc[:3,3].reshape((3,1))

    # euler angles from rotation matrix
    r, p, y = rpy_from_dcm(Cwc).reshape(3).astype(float)

    # get individual rotation matrices
    C_r = dcm_from_rpy([r,0,0])
    C_p = dcm_from_rpy([0,p,0])
    C_y = dcm_from_rpy([0,0,y])

    # skew-symmetric matrices for taking partial derivatives of rotation matrices
    S_r = np.array([
       [0, 0, 0],
       [0, 0,-1],
       [0, 1, 0],
    ])
    S_p = np.array([
       [0, 0, 1],
       [0, 0, 0],
       [-1,0, 0],
    ])
    S_y = np.array([
       [0,-1, 0],
       [1, 0, 0],
       [0, 0, 0],
    ])

    # compute partials of Cwc
    dCwc_dr = C_y @ C_p @ (S_r @ C_r)
    dCwc_dp = C_y @ (S_p @ C_p) @ C_r
    dCwc_dy = (S_y @ C_y) @ C_p @ C_r

    # pinhole projection model (from A2 instrs)
    x = K @ Cwc.T @ (Wpt - t)
    # [x_s, y_s] = x / x[2]
    # compute partials of x
    dxdtx = -K @ Cwc.T @ np.array([1,0,0])
    dxdty = -K @ Cwc.T @ np.array([0,1,0])
    dxdtz = -K @ Cwc.T @ np.array([0,0,1])
    dxdr  = K @ dCwc_dr.T @ (Wpt - t)
    dxdp  = K @ dCwc_dp.T @ (Wpt - t)
    dxdy  = K @ dCwc_dy.T @ (Wpt - t)

    # assemble Jacobian
    J = np.empty((2,6))

    def quotient_rule(f, g, df, dg):
       # derivative of f/g
       return (df*g - f*dg) / (g**2)

    for i in range(2):
        J[i,0] = quotient_rule(x[i], x[2], dxdtx[i], dxdtx[2])
        J[i,1] = quotient_rule(x[i], x[2], dxdty[i], dxdty[2])
        J[i,2] = quotient_rule(x[i], x[2], dxdtz[i], dxdtz[2])
        J[i,3] = quotient_rule(x[i], x[2], dxdr[i], dxdr[2])
        J[i,4] = quotient_rule(x[i], x[2], dxdp[i], dxdp[2])
        J[i,5] = quotient_rule(x[i], x[2], dxdy[i], dxdy[2])
        
    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J