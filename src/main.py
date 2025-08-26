# Packages import
import image_handling
import utils

# Project variables : to change for your configuration
image_dir = "content/IMG"
resized_dir = "content/IMG_resized"
max_images = 2

# Check if all directories are present.
if not utils.startup_check(image_dir, resized_dir):
    exit()

# Get a sample of images.
# Exit if samples can't be picked up.
sample_paths = utils.get_images_sample(image_dir, max_images)
if sample_paths is None:
    exit()

batch_segmentations_results, list_of_resized_images_paths = image_handling.segment_images_batch(sample_paths, image_dir, resized_dir)

for i in range(len(list_of_resized_images_paths)):
    image_handling.show_result(list_of_resized_images_paths[i], batch_segmentations_results[i])
