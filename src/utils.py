import os
import random


def startup_check(image_directory: str, resized_image_directory: str) -> bool:
    """
    Checks if the directories to run the project are present.

    Args:
        image_directory (str): Original images directory path.
        resized_image_directory (str): Directory that receive the resized images.
    Returns:
        bool: True if everything is ok to continue. False otherwise.
    """
    # Check if image directory is present
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)
        print(f"Directory '{image_directory}' created. Move your images files in.")
        return False
    else:
        print(f"Directory '{image_directory}' present.")

    # Check if resized image directory is present
    if not os.path.exists(resized_image_directory):
        os.makedirs(resized_image_directory)
        print(
            f"Directory '{resized_image_directory}' created."
        )
    else:
        print(f"Directory '{resized_image_directory}' present.")

    return True


def get_images_sample(image_directory: str, sample_size: int) -> list[str] | None:
    """
    Gets a sample of images in the image directory.

    Args:
        image_directory (str): Original images directory path.
        sample_size (int): Size of desired images.

    Returns
        list[str]: Sample of images path. None in case where the sample size is too big.
    """
    # Get all images file name
    image_paths = [image for image in os.listdir(image_directory)]
    
    # Check if the asked sample size is too big.
    if sample_size > len(image_paths):
        print("Sample size can't be greater than images file count.")
        return None
    else:
        # Pick random sample in the list of paths.
        sample_list = random.sample(image_paths, sample_size)
        # Check if images are present.
        if check_for_images(sample_list, image_directory):
            return sample_list
        else:
            return None


def check_for_images(sample_list: list[str], image_directory: str) -> bool:
    """
    Checks if images are present in the list.

    Args:
        sample_list (list): Sample of images path.
        image_directory (str): Original images directory path.

    Returns:
        bool: True if images are present. False otherwise.
    """
    if not sample_list:
        print(f"No image found in {image_directory}. Please add images.")
        return False
    else:
        print(f"{len(sample_list)} image(s) to process.")
        return True
