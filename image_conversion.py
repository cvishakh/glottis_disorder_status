import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#load the image as RGB
image_path = "leaves.jpg"  #path to image
image = Image.open(image_path).convert("RGB")
image_array = np.array(image)

#function for lightness method
def lightness_method(img_array):
    max_rgb = np.max(img_array, axis=-1)  # Max value among R, G, B channels
    min_rgb = np.min(img_array, axis=-1)  # Min value among R, G, B channels
    return (max_rgb + min_rgb) / 2  # Formula: (max + min) / 2

#function for average method
def average_method(img_array):
    return np.mean(img_array, axis=-1)  # Formula: (R + G + B) / 3

#function for luminosity method
def luminosity_method(img_array):
    return 0.2989 * img_array[..., 0] + 0.5870 * img_array[..., 1] + 0.1140 * img_array[..., 2]  #formula: 0.2989*R + 0.5870*G + 0.1140*B

#apply all methods
lightness_img = lightness_method(image_array)
avg_img = average_method(image_array)
luminosity_img = luminosity_method(image_array)

#plot original and three grayscale converted images
fig, axes = plt.subplots(1, 4, figsize=(8,4))

#original image
axes[0].imshow(image)
axes[0].set_title("Original")
axes[0].axis("off")

#lightness method
axes[1].imshow(lightness_img, cmap="gray")
axes[1].set_title("Lightness")
axes[1].axis("off")

#average method
axes[2].imshow(avg_img, cmap="gray")
axes[2].set_title("Average")
axes[2].axis("off")

#luminosity method
axes[3].imshow(luminosity_img, cmap="gray")
axes[3].set_title("Luminosity")
axes[3].axis("off")

plt.tight_layout()
plt.show()