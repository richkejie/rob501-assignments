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
    - LoG filter tuned for optimization

    """

    Id = np.empty(Il.shape)

    window = 11
    half_window = window//2

    SAD_vals = []

    # apply LoG filtering
    # hardcode filter (can be tuned):
    filter = np.array([
        [-1,  -1, -1],
        [-1, 9.5, -1],
        [-1,  -1, -1]
    ])

    # convolve() from scipy.ndimage.filters
    Il = convolve(Il, filter, mode='nearest')
    Ir = convolve(Ir, filter, mode='nearest')

    # pad images:
    Il_padded = np.pad(Il, maxd, mode='edge')
    Ir_padded = np.pad(Il, maxd, mode='edge')

    image_width = Il_padded.shape[1]

    # get bounding box limits
    xmin = bbox[0,0]
    xmax = bbox[0,1] + 1
    ymin = bbox[1,0]
    ymax = bbox[1,1] + 1

    # check possible disparities (faster computation)
    ones_filter = np.ones((half_window, half_window)) # used to sum abs_diffs through convolution
    for d in range(0, maxd+1):
        left_img = Il_padded[ymin+maxd:ymax+maxd,xmin+maxd:xmax+maxd]
        right_img = Ir_padded[ymin+maxd:ymax+maxd,xmin+maxd-d:xmax+maxd-d]
        assert(left_img.shape == right_img.shape)
        abs_diff = np.abs(left_img - right_img)
        SAD = convolve(abs_diff,ones_filter,mode='mirror')
        SAD_vals.append(SAD)

    SAD_vals = np.stack(SAD_vals, axis=2)
    disparities = np.argmin(SAD_vals, axis=2)
    Id[ymin:ymax,xmin:xmax] = disparities

    # apply additional filter for disparity smoothing (can be tuned)
    # apply only to px within bbox
    # filter from scipy.ndimage.filters
    bbox_disparities = Id[ymin:ymax,xmin:xmax]
    bbox_disparities = median_filter(
        bbox_disparities, size=13, mode='nearest'
    )
    bbox_disparities = percentile_filter(
        bbox_disparities, percentile=55, size=4, mode='nearest'
    )
    Id[ymin:ymax,xmin:xmax] = bbox_disparities

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id