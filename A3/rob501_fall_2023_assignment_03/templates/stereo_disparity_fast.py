import numpy as np
from scipy.ndimage.filters import *

def stereo_disparity_fast(Il, Ir, bbox, maxd):
    """
    Fast stereo correspondence algorithm.

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

    Id = np.empty(Il.shape)

    # window (arbitrary, can be tuned)
    window = 15
    half_window = window//2

    SAD_vals = np.empty((1,2*maxd))
    disparity_vals = np.empty((1,2*maxd))

    # pad stereo images
    Il_padded = np.pad(Il, half_window, mode='edge')
    Ir_padded = np.pad(Ir, half_window, mode='edge')

    image_width = Il_padded.shape[1]

    # get bounding box limits
    x_min = bbox[0,0]
    x_max = bbox[0,1] + 1
    y_min = bbox[1,0]
    y_max = bbox[1,1] + 1

    # find best disparities
    # steps:
    #   1) go through each px and get patch around that px of size window
    #   2) for each disparity up to maxd, find corresponding patch in other img
    #   3) compute SAD
    #   4) get min SAD to get best disparities
    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            left_patch = Il_padded[y:y+window,x:x+window]

            # find max disparity
            i = 0
            for d in range(-maxd, maxd):
                disparity_middle = x+d+half_window
                right_x_bound = image_width - half_window
                left_x_bound = half_window - 1
                if disparity_middle < right_x_bound and disparity_middle > left_x_bound:
                    right_patch = Ir_padded[y:y+window,x+d:x+d+window]
                    SAD_vals[0,i] = np.sum(np.abs(left_patch - right_patch))
                else: # out of bounds
                    SAD_vals[0,i] = -np.inf

                disparity_vals[0,i] = abs(d)
                i += 1
            
            SAD = SAD_vals[0]
            SAD[SAD<0] = np.amax(SAD) # set negative vals to max (basically don't want to worry about those)
            best = np.argmin(SAD)
            Id[y,x] = disparity_vals[0,best] # use min SADs to get best disparities

    #------------------

    correct = isinstance(Id, np.ndarray) and Id.shape == Il.shape

    if not correct:
        raise TypeError("Wrong type or size returned!")

    return Id