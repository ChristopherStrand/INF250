import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def sobel(image, threshold=80):
    Gx_kernel = np.array([[-1,0,1],
                          [-2,0,2],
                          [-1,0,1]], dtype=float)

    Gy_kernel = np.array([[-1,-2,-1],
                          [ 0, 0, 0],
                          [ 1, 2, 1]], dtype=float)

    padded = np.pad(image, 1, mode="edge")
    h, w = image.shape
    Gx = np.zeros_like(image)
    Gy = np.zeros_like(image)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            Gx[i, j] = np.sum(region * Gx_kernel)
            Gy[i, j] = np.sum(region * Gy_kernel)

    G = np.sqrt(Gx**2 + Gy**2)
    G_norm = (G / G.max()) * 255
    edges = np.where(G_norm > threshold, 255, 0).astype(np.uint8)

    plt.figure(figsize=(12,3))
    plt.subplot(1,4,1); plt.imshow(image, cmap="gray"); plt.title("Original"); plt.axis("off")
    plt.subplot(1,4,2); plt.imshow(Gx, cmap="gray"); plt.title("Sobel Gx"); plt.axis("off")
    plt.subplot(1,4,3); plt.imshow(G, cmap="gray"); plt.title("|G|"); plt.axis("off")
    plt.subplot(1,4,4); plt.imshow(edges, cmap="gray"); plt.title("Edges"); plt.axis("off")
    plt.tight_layout(); plt.show()

    return edges

image1 = np.zeros((100,100), dtype=float)
image1[20:80, 20:80] = 150
yy, xx = np.mgrid[:100, :100]
circle = (xx-50)**2 + (yy-50)**2 < 15**2
image1[circle] = 250

edges = sobel(image1, threshold=80)

filename = "images/fall.tiff"
image2 = io.imread(filename, as_gray=True)
image2 = np.array(image2, dtype=float)
image_small = image2[:400, :400]
edges2 = sobel(image_small, threshold=120)