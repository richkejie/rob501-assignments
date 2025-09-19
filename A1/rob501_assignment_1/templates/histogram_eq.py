import numpy as np

def histogram_eq(I):
    """
    Histogram equalization for greyscale image.

    Perform histogram equalization on the 8-bit greyscale intensity image I
    to produce a contrast-enhanced image J. Full details of the algorithm are
    provided in the Szeliski text.

    Parameters:
    -----------
    I  - Single-band (greyscale) intensity image, 8-bit np.array (i.e., uint8).

    Returns:
    --------
    J  - Contrast-enhanced greyscale intensity image, 8-bit np.array (i.e., uint8).
    """
    #--- FILL ME IN ---

    # Verify I is grayscale.
    if I.dtype != np.uint8:
        raise ValueError('Incorrect image format!')

    # flatten img array for convenience
    img = I.flatten() # flatten in row-major order
    N = img.shape[0] # total number of pixels

    # init J
    J = np.empty(img.shape)

    # hist calculation
    nbins = 256 # 8bit pixels (0,255) range
    rng = (0,255)
    hist, _ = np.histogram(img, bins=nbins, range=rng)
    
    # calculate cumulative dist
    PDF = hist / np.sum(hist) # normalized hist = pdf
    CDF = np.cumsum(PDF)

    # map CDF to (0,255) range
    c = np.round(255*CDF)
    # cast to uint8 for intensity
    c = c.astype('uint8')
    # f(i) = c(i)
    # pixel i of new img J will have intensity of c[img[i]]
    for i in range(N):
        J[i] = c[img[i]]
    
    # reshape J
    J = J.reshape(I.shape)

    #------------------

    return J
