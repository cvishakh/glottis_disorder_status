#import necessary libraries
import os
import random
import json
from PIL import Image
import matplotlib.pyplot as plt

#function to load images arbitartly, mask it, and extract title for "Subject disorder status"
def random_images(dataset, num_img=4):
    images, masks, status = [], [], []
    random_files = random.sample([f for f in os.listdir(dataset) if f.endswith('.png') and not f.endswith('_seg.png')], num_img)
    
    for file in random_files:
        base_name = os.path.splitext(file)[0]
        image = Image.open(os.path.join(dataset, file)).convert("RGBA")
        mask = Image.open(os.path.join(dataset, f"{base_name}_seg.png")).convert("L")
        
        #read .meta file for disorder status
        meta_path = os.path.join(dataset, f"{base_name}.meta")
        with open(meta_path) as meta_file:
            meta_data = json.load(meta_file)
            disorder_status = meta_data.get("Subject disorder status", "Unknown")

        images.append(image)
        masks.append(mask)
        status.append(disorder_status)

    return images, masks, status

#function to overlay mask on image
def mask_on_image(image, mask):
    red_mask = Image.new("RGBA", image.size, (255, 0, 0, 100))
    red_mask.putdata([(255, 0, 0, 100) if pixel > 0 else (0, 0, 0, 0) for pixel in mask.getdata()])
    return Image.alpha_composite(image, red_mask)

#path to dataset and loading images
dataset = "Mini_BAGLS_dataset"
images, masks, status = random_images(dataset)

#plot random 4 images as inference
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, image, mask, status in zip(axes.ravel(), images, masks, status):
    ax.imshow(mask_on_image(image, mask))
    ax.set_title(status)  #displays the disorder status for respective images
    ax.axis('off')

plt.tight_layout()
plt.show()