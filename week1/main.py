import matplotlib.pyplot as plt  
from skimage import io

filename = "images/fall.tiff"
fall = io.imread(filename)

plt.imshow(fall)

print(fall.shape)

fall_red = fall[:, :, 0]
print(f"max: {max(fall_red.flatten())}")
print(f"min: {min(fall_red.flatten())}")

plt.imshow(fall_red, vmin=0, vmax=255)
plt.show()


