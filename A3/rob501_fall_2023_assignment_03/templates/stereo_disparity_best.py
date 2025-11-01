import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_best(Il, Ir, bbox, maxd):
    """
    Best stereo correspondence algorithm.

    This function computes a stereo disparity image from left stereo 
    image Il and right stereo image Ir. Only disparity values within
    the bounding box region (inclusive) are evaluated.

    Parameters:
    -----------
    Il    - Left stereo image, m x n pixel np.array, greyscale.
    Ir    - Right stereo image, m x n pixel np.array, greyscale.
    bbox  - 2x2 np.array, bounding box, relative to left image, from top left
            corner to bottom right corner (inclusive). (x, y) in columns.
    maxd  - Integer, maximum disparity value; disparities must be within zero
            to maxd inclusive (i.e., don't search beyond maxd).

    Returns:
    --------
    Id  - Disparity image (map) as np.array, same size as Il.
    """
    # Hints:
    #
    #  - Loop over each image row, computing the local similarity measure, then
    #    aggregate. At the border, you may replicate edge pixels, or just avoid
    #    using values outside of the image.
    #
    #  - You may hard-code any parameters you require in this function.
    #
    #  - Use whatever window size you think might be suitable.
    #
    #  - Optimize for runtime AND for clarity.

    #--- FILL ME IN ---

    # Code goes here...

    """
    My algorithm uses ideas from this paper:
    "A stereo machine for video-rate dense depth mapping and its new applications"
    T. Kanade
    https://www.ri.cmu.edu/pub_files/pub2/kanade_takeo_1996_2/kanade_takeo_1996_2.pdf

    The paper provides the following theory:
    1) apply Laplacian of Gaussian (LoG) filter to input images
    2) compute SSD (sum of squared difference) values wrt inverse distance
        for all stereo image pairs --> SSSD: sum of SSD values
    3) find minimum of SSSD function

    I make the following changes/simplifications:
    - use SAD instead of SSD (for faster computation)
    - LoG filter tuned for optimization (use a discrete approximation)
    - apply further filtering to smooth disparity map

    """

    Id = np.empty(Il.shape)

    # window size can be tuned
    window = 10
    half_window = window//2

    # apply LoG filtering (discrete approx)
    # hardcode filter (can be tuned):
    LoG_filter = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  1,  1, -1],
        [-1,  1, 12,  1, -1],
        [-1,  1,  1,  1, -1],
        [-1, -1, -1, -1, -1]
    ])

    # convolve() from scipy.ndimage.filters
    Il = convolve(Il, LoG_filter, mode='nearest')
    Ir = convolve(Ir, LoG_filter, mode='nearest')

    # pad images:
    Il_padded = np.pad(Il, maxd, mode='edge')
    Ir_padded = np.pad(Ir, maxd, mode='edge')

    # get bounding box limits
    x_min = bbox[0,0]
    x_max = bbox[0,1] + 1
    y_min = bbox[1,0]
    y_max = bbox[1,1] + 1

    # check possible disparities (faster computation):
    # steps:
    #   1) for each disparity from 0 - maxd, get one patch from left and right imgs
    #           pick patch at maxd, with x varying with disparity being checked
    #           (patch doesn't really matter as long as its within bbox)
    #   2) compute SAD
    #   3) get min SAD to get best disparities
    SAD_vals = []
    ones_filter = np.ones((half_window, half_window)) # used to sum abs_diffs through convolution
    for d in range(0, maxd+1):
        left_patch = Il_padded[y_min+maxd:y_max+maxd,x_min+maxd:x_max+maxd]
        right_patch = Ir_padded[y_min+maxd:y_max+maxd,x_min+maxd-d:x_max+maxd-d]
        assert(left_patch.shape == right_patch.shape)
        abs_diff = np.abs(left_patch - right_patch)
        SAD = convolve(abs_diff,ones_filter,mode='mirror')
        SAD_vals.append(SAD)

    # get min SADs to get best disparities
    disparities = np.argmin(np.stack(SAD_vals, axis=2), axis=2)
    Id[y_min:y_max,x_min:x_max] = disparities

    # apply additional filter for disparity smoothing (can be tuned)
    # smoothing reduces E_rms
    # apply only to px within bbox
    # filter from scipy.ndimage.filters
    bbox_disparities = Id[y_min:y_max,x_min:x_max]
    # median filter removes noise
    bbox_disparities = median_filter(
        bbox_disparities,size=18,mode='nearest'
    )
    Id[y_min:y_max,x_min:x_max] = bbox_disparities

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id