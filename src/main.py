# Packages import
import image_handling
import utils

# Project variables : to change for your configuration
image_dir = "content/IMG"
mask_dir = "content/Mask"
resized_images_dir = "content/IMG_resized"
resized_masks_dir = "content/Mask_resized"

max_images = 5

# Check if all directories are present.
if not utils.startup_check(image_dir, resized_images_dir, mask_dir, resized_masks_dir):
    exit()

# Get a sample of images.
# Exit if samples can't be picked up.
# ___________CODE USED WHILE TESTING
# sample_paths = utils.get_images_sample(image_dir, max_images)
# if sample_paths is None:
#    exit()

images_paths, masks_paths = utils.get_all_images_and_masks(image_dir, mask_dir)

batch_segmentations_results, list_of_resized_images_paths = (
    image_handling.segment_images_batch(
        images_paths[:max_images], image_dir, resized_images_dir
    )
)

list_of_resized_masks_paths = []
for mask in masks_paths[:max_images]:
    list_of_resized_masks_paths.append(
        image_handling.resize_image(mask, mask_dir, resized_masks_dir)
    )

for i in range(len(list_of_resized_images_paths)):
    image_handling.show_result(
        list_of_resized_images_paths[i],
        list_of_resized_masks_paths[i],
        batch_segmentations_results[i],
    )
