from skimage import io, exposure, filters, util
import matplotlib.pyplot as plt

img = io.imread("images/airfield.tif")
img = util.img_as_float(img) 
fig, axes = plt.subplots(3, 3, figsize=(12, 10))
ax = axes.ravel()



#1. display the image
ax[0].imshow(img, cmap="gray")
ax[0].set_title("Original")

#2. compute the histogram of the image and plot it
ax[1].hist(img.flatten())
ax[1].set_title("Histogram (original)")

#3. increases the contrast
# https://scikit-image.org/docs/0.25.x/api/skimage.exposure.html
contrast = exposure.adjust_sigmoid(img, cutoff=0.5, gain=10)
ax[2].imshow(contrast, cmap="gray")
ax[2].set_title("Increased contrast")

#increases the light intensity
bright = exposure.adjust_gamma(img, gamma=0.7)
ax[3].imshow(bright, cmap="gray")
ax[3].set_title("Increased brightness")

#4. histogram equalisation
# https://scikit-image.org/docs/0.25.x/auto_examples/color_exposure/plot_equalize.html
# enhances contrast and redistributes the most normal intensity values.
eq = exposure.equalize_hist(img)
ax[4].imshow(eq, cmap="gray")
ax[4].set_title("Histogram equalisation")

#5. plots the histogram of the corrected image
ax[5].hist(eq.flatten())
ax[5].set_title("histogram equalisation")

#6. makes a thresholding (otsu) of the image
# https://scikit-image.org/docs/0.25.x/auto_examples/segmentation/plot_thresholding.html
otsu = img > filters.threshold_otsu(img)
ax[6].imshow(otsu, cmap="gray")
ax[6].set_title("Otsu thresholding")

ax[7].axis("off")
ax[8].axis("off")

plt.tight_layout()
plt.show()






#hint otsu:
#https://www.baeldung.com/cs/otsu-segmentation
