import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import albumentations as A

#Set random seed using matriculation number
np.random.seed(23240372)  # Replace with your matriculation number

#Define the folder containing the images and masks
data_dir = r".\Mini_BAGLS_dataset"  # Adjust the path to your dataset

#List all files in the directory
all_files = sorted(os.listdir(data_dir))

#Separate images and masks based on naming convention
image_files = [f for f in all_files if not f.endswith("_seg.png")]
mask_files = [f for f in all_files if f.endswith("_seg.png")]

#Create a dictionary to pair images with masks based on the common identifier
image_mask_pairs = {}
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]  # Extract the base name, e.g., "20"
    mask_file = f"{base_name}_seg.png"
    if mask_file in mask_files:
        image_mask_pairs[image_file] = mask_file

#Randomly select an image-mask pair
random_image = np.random.choice(list(image_mask_pairs.keys()))
random_mask = image_mask_pairs[random_image]

#Load the selected image and mask
image_path = os.path.join(data_dir, random_image)
mask_path = os.path.join(data_dir, random_mask)

image = cv2.imread(image_path)
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

#Check if images are loaded properly
if image is None:
    raise ValueError(f"Failed to load image from: {image_path}")

if mask is None:
    raise ValueError(f"Failed to load mask from: {mask_path}")

#Convert image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#Define an augmentation pipeline with titles
augmentations = [
    (A.HorizontalFlip(p=0.5), "Horizontal Flip"),
    (A.RandomBrightnessContrast(p=0.2), "Random Contrast"),
    (A.Rotate(limit=30, p=1), "Rotate 30°"),
    (A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3), "Elastic Transform")
]

#Apply augmentations to the image and mask
augmented_images = []
augmented_masks = []
titles = []

for aug, title in augmentations:
    augmented = aug(image=image, mask=mask)
    augmented_images.append(augmented['image'])
    augmented_masks.append(augmented['mask'])
    titles.append(title)

#Plot original and augmented images with masks
plt.figure(figsize=(15, 10))

#Plot the original
plt.subplot(3, 4, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis("off")

plt.subplot(3, 4, 2)
plt.title("Original Mask")
plt.imshow(mask, cmap="gray")
plt.axis("off")

#Plot augmented images and masks with titles
for i, title in enumerate(titles):
    plt.subplot(3, 4, 3 + i * 2)
    plt.title(title)
    plt.imshow(augmented_images[i])
    plt.axis("off")
    
    plt.subplot(3, 4, 4 + i * 2)
    plt.title(f"{title} (Mask)")
    plt.imshow(augmented_masks[i], cmap="gray")
    plt.axis("off")

plt.tight_layout()
plt.show()