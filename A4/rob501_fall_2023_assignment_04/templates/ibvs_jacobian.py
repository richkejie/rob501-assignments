import numpy as np

def ibvs_jacobian(K, pt, z):
    """
    Determine the Jacobian for IBVS.

    The function computes the Jacobian for image-based visual servoing,
    given the camera matrix K, an image plane point, and the estimated
    depth of the point. The x and y focal lengths in K are guaranteed 
    to be identical.

    Parameters:
    -----------
    K  - 3x3 np.array, camera intrinsic calibration matrix.
    pt - 2x1 np.array, image plane point. 
    z  - Scalar depth value (estimated).

    Returns:
    --------
    J  - 2x6 np.array, Jacobian. The matrix must contain float64 values.
    """

    #--- FILL ME IN ---

    J = np.zeros((2,6))

    # extract variables from K
    focal_length = K[0,0]
    c_x = K[0,2]
    c_y = K[1,2]

    # normalized image plane coordinates
    ubar = pt[0] - c_x
    vbar = pt[1] - c_y

    # calculate Jacobian based on equation from Corke text
    J[0,0] = -focal_length / z
    J[0,2] = ubar / z
    J[0,3] = ubar * vbar / focal_length
    J[0,4] = -(focal_length**2 + ubar**2) / focal_length
    J[0,5] = vbar

    J[1,1] = -focal_length / z
    J[1,2] = vbar / z
    J[1,3] = (focal_length**2 + vbar**2) / focal_length
    J[1,4] = -ubar * vbar / focal_length
    J[1,5] = -ubar

    #------------------

    correct = isinstance(J, np.ndarray) and \
        J.dtype == np.float64 and J.shape == (2, 6)

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return J