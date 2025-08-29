import os
import random


def startup_check(
    image_directory: str,
    resized_image_directory: str,
    mask_directory: str,
    resized_mask_directory: str,
) -> bool:
    """
    Checks if the directories to run the project are present.

    Args:
        image_directory (str): Original images directory path.
        resized_image_directory (str): Directory that receive the resized images.
        mask_directory (str): Original mask directory path.
        resized_mask_directory (str): Directory that receive the resized images.
    Returns:
        bool: True if everything is ok to continue. False otherwise.
    """
    # Check if all directories are present
    if not (
        check_dir(image_directory, is_resized_dir=False)
        and check_dir(mask_directory, is_resized_dir=False)
        and check_dir(resized_image_directory, is_resized_dir=True)
        and check_dir(resized_mask_directory, is_resized_dir=True)
    ):
        return False
    else:
        return True


def check_dir(dir_path: str, is_resized_dir: bool) -> bool:
    """
    Checks if the given directory is present. Creates it otherwise.
    Handles resized directory differently.

    Args:
        dir_path (str): Directory path to check.
        is_resized_dir (bool): True if it is an resized directory. False otherwise.
    
    Returns:
        bool: True in all cases in case of a resized directory. In case of an original images directory, True it is present, False otherwise.
    """
    # Checks if the directory exist.
    if not os.path.exists(dir_path):
        # If not, creates it.
        os.makedirs(dir_path)
        if is_resized_dir:
            # If it's a resized dir, the script can work without problem.
            print(f"Directory '{dir_path}' created.")
            return True
        else:
            # In case of a original image directory, alert the user to populate the directory with files.
            print(f"Directory '{dir_path}' created. Move your images files in.")
            return False
    else:
        # Tell the user that the directory is present and ready to go.
        print(f"Directory '{dir_path}' present.")
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

def get_all_images_and_masks(image_directory: str, mask_directory: str) -> tuple[list, list]:
    """
    Gets all images available and their corresponding masks.

    Args:
        image_directory (str): Original images directory path.
        mask_directory (str): Original mask directory path.
    
    Returns
        tuple[list, list]: Returns a tuple with all images paths, and masks paths. Return None if no image found.
    """

    image_paths = [image for image in os.listdir(image_directory)]
    masks_paths = [mask for mask in os.listdir(mask_directory)]

    if check_for_images(image_paths, image_directory) and check_for_images(masks_paths, mask_directory):
        return image_paths, masks_paths
    else:
        return None