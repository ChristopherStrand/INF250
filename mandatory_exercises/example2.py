# -*- coding: utf-8 -*-
"""
Skeleton for first part of the blob-detection coursework as part of INF250
at NMBU (Autumn 2017).
"""
__author__ = "Yngve Mardal Moe"
__email__ = "yngve.m.moe@gmail.com"

import numpy as np
import matplotlib.pyplot as plt
from skimage import io


def threshold(image, th=None):
    """Returns a binarised version of given image, thresholded at given value.
    Binarises the image using a global threshold `th`. Uses Otsu's method
    to find optimal thrshold value if the threshold variable is None. The
    returned image will be in the form of an 8-bit unsigned integer array
    with 255 as white and 0 as black.
    Parameters:
    -----------
    image : np.ndarray
    Image to binarise. If this image is a colour image then the last
    dimension will be the colour value (as RGB values).
    th : numeric
    Threshold value. Uses Otsu's method if this variable is None.
    Returns:
    --------
    binarised : np.ndarray(dtype=np.uint8)
    Image where all pixel values are either 0 or 255.
    """
    # Setup
    shape = np.shape(image)
    binarised = np.zeros([shape[0], shape[1]], dtype=np.uint8)
    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')
    if th is None:
        th = otsu(image)
    binarised[image >= th] = 255
    binarised[image < th] = 0
    return binarised


def histogram(image):
    """Returns the image histogram with 256 bins.
    """
    # Setup
    shape = np.shape(image)
    histogram = np.zeros(256)
    if len(shape) == 3:
        image = image.mean(axis=2)
    elif len(shape) > 3:
        raise ValueError('Must be at 2D image')
    K = 256
    histogram = np.zeros(K)
    for i in range(shape[0]):
        for j in range(shape[1]):
            pixval = int(image[i, j])
            histogram[pixval] += 1
    return histogram


def otsu(image):
    """
    Finds the optimal threshold value of a given grayscale image using Otsu's method.

    This implementation computes the between-class variance efficiently by using
    cumulative sums rather than iterating over all possible threshold values.

    Steps:
    -------
    1. Compute the histogram of the image (0–255) and normalize it into probabilities.
    2. Calculate cumulative sums:
        - P[t]: cumulative probability up to intensity t (background proportion)
        - S[t]: cumulative weighted intensity sum up to t
    3. The total mean intensity (mu_T) is taken as S[-1].
    4. For every possible threshold t, compute the between-class variance:
           var_between[t] = ((mu_T * P[t] - S[t]) ** 2) / (P[t] * (1 - P[t]))
       This measures how well the image is separated into two classes
       (background and foreground) at threshold t.
    5. The threshold that maximizes var_between is chosen as the optimal one.

    This vectorized implementation avoids explicit loops and computes the 
    optimal threshold in O(256) time using NumPy array operations.

    Parameters:
    -----------
    image : np.ndarray
        Grayscale (or RGB) image array. If RGB, it should be averaged before input.

    Returns:
    --------
    th : int
        Optimal global threshold value (0–255) computed using Otsu's method.
    """
    hist = histogram(image).astype(float)
    total = hist.sum()
    if total == 0:
        return 0

    p = hist / total
    intensity = np.arange(256, dtype=float)
    P = np.cumsum(p)
    S = np.cumsum(p * intensity)
    mu_T = S[-1]
    eps = 1e-12
    var_between = (mu_T * P - S) ** 2 / (P * (1.0 - P) + eps)
    var_between[0] = 0.0
    var_between[-1] = 0.0
    th = int(np.argmax(var_between))
    return th


image = "gingerbreads.jpg"
gingerbreads = io.imread(image)
print(f"Optimal threshold using Otsu's method: {otsu(gingerbreads)}")
plt.figure()
plt.plot(histogram(gingerbreads))
plt.figure()
# plt.plot(threshold(gingerbreads))
io.imshow(threshold(gingerbreads))