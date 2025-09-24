# Billboard hack script file.
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.path import Path
from imageio import imread, imwrite

from dlt_homography import dlt_homography
from bilinear_interp import bilinear_interp
from histogram_eq import histogram_eq

def billboard_hack():
    """
    Hack and replace the billboard!

    Parameters:
    ----------- 

    Returns:
    --------
    Ihack  - Hacked RGB intensity image, 8-bit np.array (i.e., uint8).
    """
    # Bounding box in Y & D Square image.
    bbox = np.array([[404, 490, 404, 490], [38,  38, 354, 354]])

    # Point correspondences.
    Iyd_pts = np.array([[416, 485, 488, 410], [40,  61, 353, 349]])
    Ist_pts = np.array([[2, 218, 218, 2], [2, 2, 409, 409]])

    Iyd = imread('../billboard/yonge_dundas_square.jpg')
    Ist = imread('../billboard/uoft_soldiers_tower_dark.png')

    Ihack = np.asarray(Iyd)
    Ist = np.asarray(Ist)

    #--- FILL ME IN ---

    # Let's do the histogram equalization first.
    st_histeq = histogram_eq(Ist) # hist equalized soldiers tower image

    # Compute the perspective homography we need...
    H, A = dlt_homography(Iyd_pts, Ist_pts)

    # Main 'for' loop to do the warp and insertion - 
    # this could be vectorized to be faster if needed!

    # You may wish to make use of the contains_points() method
    # available in the matplotlib.path.Path class!
    billboard_yd = Path(Iyd_pts.T)
    bbox_x_min = np.min(bbox[0])
    bbox_x_max = np.max(bbox[0])
    bbox_y_min = np.min(bbox[1])
    bbox_y_max = np.max(bbox[1])
    for x in range(bbox_x_min, bbox_x_max+1):
        for y in range(bbox_y_min, bbox_y_max+1):
            
            if billboard_yd.contains_points(np.array([[x,y]])):
                # apply homography
                P_yd = np.array([x,y,1])
                P_st = np.matmul(H, P_yd)
                P_st = P_st/P_st[-1] # normalize by last element

                # apply bilinear interp of histogram equalized img
                P_st_pt = P_st[:-1].reshape((2,1))

                i_val = bilinear_interp(st_histeq, P_st_pt)
                Ihack[y,x] = np.array([i_val, i_val, i_val]) # rgb values
            else:
                continue

    #------------------

    # plt.imshow(Ihack)
    # plt.show()
    # imwrite(Ihack, 'billboard_hacked.png')

    return Ihack

# billboard_hack()