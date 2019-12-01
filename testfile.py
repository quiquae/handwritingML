from skimage.filters import try_all_threshold
from skimage.data import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
img = rgb2gray(imread("data/actual_image13.jpg"))
fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
plt.show()