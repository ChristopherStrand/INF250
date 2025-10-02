# -*- coding: utf-8 -*-

"""
Exercise on edge operators and sharpening with skimage
"""

import numpy as np
from skimage import color, img_as_float
from skimage.filters import sobel, prewitt, laplace, unsharp_mask
from skimage.feature import canny
from skimage import io


def edge_operator(image, operator):
    """Returns the result from one of the edge operators:
    1 = sobel filter
    2 = prewitt filter
    3 = canny filter
    4 = laplace filter
    """
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
    image = img_as_float(image)

    if operator == 1:
        filtered = sobel(image)
    elif operator == 2:
        filtered = prewitt(image)
    elif operator == 3:
        filtered = canny(image, sigma=1.0).astype(float)
    elif operator == 4:
        filtered = laplace(image)
    else:
        raise ValueError("operator må være 1 (sobel), 2 (prewitt), 3 (canny) eller 4 (laplace)")

    return filtered


def sharpen(image, sharpmask):
    """Performs an image sharpening:
    1 = Laplace
    2 = USM
    """
    if image.ndim == 3:
        if image.shape[-1] == 4:      
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)
    image = img_as_float(image)

    if sharpmask == 1:
        alpha = 0.5
        sharpened = image - alpha * laplace(image)
    elif sharpmask == 2:
        sharpened = unsharp_mask(image, radius=1.5, amount=1.5, preserve_range=True)
    else:
        raise ValueError("sharpmask må være 1 (Laplace) eller 2 (USM)")

    return np.clip(sharpened, 0, 1)



img = io.imread("images/AthenIR.png")

edges = edge_operator(img, 1)       
sharp = sharpen(img, 2)     

io.imsave("week4/edges.png", (edges*255).astype(np.uint8))
io.imsave("week4/sharp.png", (sharp*255).astype(np.uint8))