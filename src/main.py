# Packages import
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from dotenv import load_dotenv
import image_handling
from pathlib import Path
from PIL import Image
import numpy as np

# Loading .env file
load_dotenv()

# Project variables : to change for your configuration
image_dir = "content/IMG"
max_images = 3

api_token = os.getenv("HUGGINGFACE_API_TOKEN")

# Check if image directory is present
if not os.path.exists(image_dir):
    os.makedirs(image_dir)
    print(f"Directory '{image_dir}' created. Move your images files in.")
    exit()
else:
    print(f"Directory '{image_dir}' present.")

# Get all images file name
image_paths = [image for image in os.listdir(image_dir)]

# Check if the images are present
if not image_paths:
    print(f"No image found in {image_dir}. Please add images.")
    exit()
else:
    print(f"{len(image_paths)} image(s) to process.")

# For now, get the data and the extension of a single file
# First change max size
image_path = Path(image_dir) / image_paths[0]
resized_image = image_handling.resize_image(image_path)
# Get file data
data, file_extension = image_handling.read_image(resized_image)

# API setup
API_URL = (
    "https://router.huggingface.co/hf-inference/models/sayeed99/segformer_b3_clothes"
)
headers = {
    "Authorization": f"Bearer {api_token}",
    "Content-Type": f"image/{file_extension}",
}

# Get the API result
api_result = image_handling.segmentation_query(data, API_URL, headers)
image_width, image_height = image_handling.get_image_dimensions(resized_image)
combined_masks = image_handling.create_masks(api_result, image_width, image_height)

with Image.open(resized_image) as image:
    plt.figure(figsize=(15, 5))

    # Position 1
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original image")
    plt.axis("off")

    # Position 2
    combined_masks_wo_background = np.ma.masked_where(combined_masks == 0, combined_masks)
    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(combined_masks_wo_background)
    plt.title("Image masks with original background")
    plt.axis("off")

    # Position 3
    plt.subplot(1, 3, 3)
    plt.imshow(combined_masks)
    plt.title("Image masks")
    plt.axis("off")

    plt.tight_layout()
    plt.show()