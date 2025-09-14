import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters

img = io.imread("images/fall.tiff")
gray = color.rgb2gray(img)



gauss_sigma2 = filters.gaussian(gray, sigma=2)
gauss_sigma10 = filters.gaussian(gray, sigma=10)

plt.subplot(1, 3, 1); plt.imshow(gray, cmap="gray"); plt.title("Original"); plt.axis("off")
plt.subplot(1, 3, 2); plt.imshow(gauss_sigma2, cmap="gray"); plt.title("Gaussian σ=2"); plt.axis("off")
plt.subplot(1, 3, 3); plt.imshow(gauss_sigma10, cmap="gray"); plt.title("Gaussian σ=10"); plt.axis("off")
plt.show()

plt.hist(gray.ravel(), 256, color="black", alpha=0.5, label="Original")
plt.hist(gauss_sigma2.ravel(), 256, color="blue", alpha=0.5, label="σ=2")
plt.hist(gauss_sigma10.ravel(), 256, color="red", alpha=0.5, label="σ=10")
plt.legend(); plt.show()


""" 
This it the part to understand mean, median, ... filter


"""
test_img = np.array([
    [255, 255, 255, 255, 255],
    [255,   0,   0,   0, 255],
    [  0,   0,   0,   0,   0],
    [255,   0,   0,   0, 255],
    [255, 255, 255, 255, 255]
], dtype=np.float32)

k = 3
pad = k // 2
padded = np.pad(test_img, pad_width=pad, mode='constant', constant_values=0)
output = np.zeros_like(test_img)

for v in range(test_img.shape[0]):
    for u in range(test_img.shape[1]):
        region = padded[v:v+k, u:u+k]
        output[v, u] = region.mean()

print("Original:\n", test_img.astype(int))
print("\nmean filter:\n", output.astype(int))
