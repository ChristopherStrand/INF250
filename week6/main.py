from skimage import io, img_as_float
from skimage.color import rgb2hsv, hsv2rgb, rgb2lab, lab2rgb
import matplotlib.pyplot as plt
import numpy as np


img = io.imread("images/wheat.jpg")
img = img_as_float(img)

R = img[..., 0]
G = img[..., 1]
B = img[..., 2]

lab = rgb2lab(img)
L = lab[..., 0]   
a = lab[..., 1]  
b = lab[..., 2]   

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

axes[0, 0].imshow(img)
axes[0, 0].set_title("Original RGB")
axes[0, 1].imshow(R, cmap="gray")
axes[0, 1].set_title("R channel")
axes[0, 2].imshow(G, cmap="gray")
axes[0, 2].set_title("G channel")
axes[1, 0].imshow(B, cmap="gray")
axes[1, 0].set_title("B channel")
axes[1, 1].imshow(a, cmap="gray")
axes[1, 1].set_title("Lab: a channel")
axes[1, 1].axis("off")
axes[1, 2].imshow(b, cmap="gray")
axes[1, 2].set_title("Lab: b channel")
plt.tight_layout()
plt.show()



img_2 = io.imread("images/IMG_3223.png")
img_2 = img_as_float(img_2) 


hsv = rgb2hsv(img_2)

hsv_v1 = hsv.copy()
hsv_v1[..., 2] = 1.0
img_v1 = hsv2rgb(hsv_v1)
S = hsv[..., 1]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_2)
axes[0].set_title("Original")
axes[1].imshow(np.clip(img_v1, 0, 1))
axes[1].set_title("V = 1 (normalized brightness)")
axes[2].imshow(S, cmap="gray")
axes[2].set_title("Saturation (S)")

plt.tight_layout()
plt.show()