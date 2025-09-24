import numpy as np
from numpy.linalg import inv

def linear_interp(x, x1, y1, x2, y2):
    y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
    return int(round(y))

def bilinear_interp(I, pt):
    """
    Performs bilinear interpolation for a given image point.

    Given the (x, y) location of a point in an input image, use the surrounding
    4 pixels to conmpute the bilinearly-interpolated output pixel intensity.

    Note that images are (usually) integer-valued functions (in 2D), therefore
    the intensity value you return must be an integer (use round()).

    This function is for a *single* image band only - for RGB images, you will 
    need to call the function once for each colour channel.

    Parameters:
    -----------
    I   - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).
    pt  - 2x1 np.array of point in input image (x, y), with subpixel precision.

    Returns:
    --------
    b  - Interpolated brightness or intensity value (whole number >= 0).
    """
    #--- FILL ME IN ---

    if pt.shape != (2, 1):
        raise ValueError('Point size is incorrect.')

    Px = pt[0]
    Py = pt[1]

    y_min = 0
    y_max = I.shape[0] - 1
    x_min = 0
    x_max = I.shape[1] - 1

    # get 4 points (x,y) pixel coords from I
    x_1 = int(np.floor(Px))
    if (x_1 - Px) == 0: # Px int
        x_1 -= 1

    x_2 = int(x_1 + 1)

    y_1 = int(np.floor(Py))
    if (y_1- Py) == 0: # Py int
        y_1 -= 1

    y_2 = int(y_1 + 1)

    # check if on edge and need to do linear interpolation
    if y_1 <= y_min:
        y_star = y_min
    elif y_2 >= y_max:
        y_star = y_max
    else:
        y_star = None
    
    if y_star != None:
        return linear_interp(
            x = Px[0],
            x1 = x_1,
            y1 = I[y_star][x_1],
            x2 = x_2,
            y2 = I[y_star][x_2]
        )

    if x_1 <= x_min:
        x_star = x_min
    elif x_2 >= x_max:
        x_star = x_max
    else:
        x_star = None

    if x_star != None:
        return linear_interp(
            x = Py[0],
            x1 = y_1,
            y1 = I[y_1][x_star],
            x2 = y_2,
            y2 = I[y_2][x_star]
        )

    # 4 points are:
    # Q11(x_1, y_1)
    # Q12(x_1, y_2)
    # Q21(x_2, y_1)
    # Q22(x_2, y_2)
    # get intensities at these 4 points:
    # note: I is y by x (since y represents the rows)
    I_Q11 = I[y_1][x_1]
    I_Q12 = I[y_2][x_1]
    I_Q21 = I[y_1][x_2]
    I_Q22 = I[y_2][x_2]

    # get multilinear polynomial system
    # derived from Wikipedia: https://en.wikipedia.org/wiki/Bilinear_interpolation
    M = np.array([
        [1, x_1, y_1, x_1*y_1],
        [1, x_1, y_2, x_1*y_2],
        [1, x_2, y_1, x_2*y_1],
        [1, x_2, y_2, x_2*y_2],
    ])
    fI = np.array([
        I_Q11,
        I_Q12,
        I_Q21,
        I_Q22,
    ])
    A = np.matmul(inv(M), fI) # solve matrix linear equation

    # calculate b from elements of A
    a_1 = A[0]
    a_2 = A[1]
    a_3 = A[2]
    a_4 = A[3]
    res = a_1 + a_2*Px + a_3*Py + a_4*Px*Py
    b = int(round(res[0])) # intensity in uint8

    #------------------

    return b
